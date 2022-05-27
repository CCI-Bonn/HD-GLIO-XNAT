#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:AUTHOR: Jens Petersen
:ORGANIZATION: Heidelberg University Hospital; German Cancer Research Center
:CONTACT: jens.petersen@dkfz.de
:SINCE: Wed May 17 14:29:20 2017
:VERSION: 0.1

DESCRIPTION
-----------

Update all scan types under a given XNAT endpoint

REQUIRES
--------

pydicom
pyxnat

TODO
----



"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

import argparse
import os
import re
import time
import traceback

import pydicom as dicom
import pyxnat
import pyxnat.core.resources as res

# =============================================================================
# PROGRAM METADATA
# =============================================================================

__author__ = "Jens Petersen"
__email__ = "jens.petersen@dkfz.de"
__copyright__ = ""
__license__ = ""
__date__ = "Wed May 17 14:29:20 2017"
__version__ = "0.1"

# Edited by Hagen Meredig Wed May 25 2022

RESERVED_RANGE = ("8999", "9990", "9991", "9992", "9993", "9994", "9995")

# =============================================================================
# METHODS & CLASSES
# =============================================================================


def value_from_search(searches, dicom_string, hard_match=False):
    """Extract value for searches from pydicom representation of header"""

    if hasattr(searches, "__iter__") and not isinstance(searches, str):

        result_dict = {}

        for i, search in enumerate(searches):
            if hasattr(hard_match, "__iter__"):
                result_dict.update(
                    value_from_search(search, dicom_string, hard_match[i])
                )
            else:
                result_dict.update(value_from_search(search, dicom_string, hard_match))

    else:

        key_start = "\(\w{4}\,\s\w{4}\)"
        value_start = "[A-Z]{2}\:\s"

        if not hard_match:
            search_string = (
                key_start + ".*" + searches + ".*" + value_start + ".*" + "\n"
            )
        else:
            search_string = (
                key_start + "\s.?" + searches + ".?\s+" + value_start + ".*" + "\n"
            )

        pattern = re.compile(search_string)
        results = re.findall(pattern, dicom_string)
        result_dict = {}

        for result in results:

            entry, value = re.split(value_start, result)
            value = value.strip()
            if value.startswith("'"):
                value = value[1:]
            if value.endswith("'"):
                value = value[:-1]
            value = value.strip()
            entry = re.split(key_start, entry)[1].strip()
            #            entry = re.findall("[\w\s]+", entry)[0]
            result_dict[entry] = value

    return result_dict


def extract_contrast(dicom_data):

    contrast = "unknown"

    info = value_from_search(
        [
            "Acquisition Contrast",
            "Series Description",
            "Sequence Name",
            "Protocol Name",
            "Inversion Recovery",
            "Contrast Bolus Agent",
            "Image Type",
        ],
        dicom_data.__repr__(),
        True,
    )

    # first check for acquisition contrast
    if "Acquisition Contrast" in info:
        if info["Acquisition Contrast"] == "FLUID_ATTENUATED":
            contrast = "FLAIR"
        elif info["Acquisition Contrast"] == "T2":
            contrast = "T2"
        elif info["Acquisition Contrast"] == "T1":
            contrast = "T1"

    # check for FLAIR occurrences
    if contrast in ("unknown", "T2"):
        for key in info:
            if "flair" in info[key].lower():
                contrast = "FLAIR"

    # T2 and inversion recovery will also usually mean FLAIR
    if contrast == "T2":
        if "Inversion Recovery" in info:
            if info["Inversion Recovery"].lower() == "yes":
                contrast = "FLAIR"

    # check if 3D FLAIR
    if contrast == "FLAIR":
        for key in info:
            if "3d" in info[key].lower():
                contrast = "FLAIR3D"

    # if still unknown, count occurrences of T1 and T2
    if contrast == "unknown":
        count_T1 = 0
        count_T2 = 0
        for key in info:
            count_T1 += info[key].lower().count("t1")
            count_T2 += info[key].lower().count("t2")
        if count_T1 > count_T2:
            contrast = "T1"
        elif count_T2 > count_T1:
            contrast = "T2"

    # check for contrast agent in T1
    if contrast == "T1":
        for indicator in ("km", "post", "gad", "contrast", "gd", "+c", "kontrast"):
            for key in info:
                if indicator in info[key].lower():
                    contrast = "T1ce"
        for indicator in ("pre", "prae"):
            for key in info:
                if indicator in info[key].lower():
                    contrast = "T1"

    # check if subtraction
    if contrast in ("T1", "T1ce"):
        for key in info:
            if "sub" in info[key].lower():
                contrast = "T1sub"

    # check for SWI
    for key in info:
        if "swi" in info[key].lower():
            contrast = "SWI"

    # check for ADC
    for key in info:
        if "adc" in info[key].lower():
            contrast = "ADC"

    # check for CBV
    gre_epi_perf = False
    relcbv = False
    rgb = False
    for key in info:
        if "gre_epi_perf" in info[key].lower():
            gre_epi_perf = True
        if "relcbv" in info[key].lower():
            relcbv = True
        if "rgb" in info[key].lower():
            rgb = True
    if gre_epi_perf and relcbv and not rgb:
        contrast = "CBV"

    return contrast


def process_single(xnat_object, wait=1, verbose=True, raise_=False):

    try:

        if verbose:
            print("Found scan: {},".format(xnat_object._uri))

        scan_number = xnat_object.attrs.get("ID")
        if scan_number in RESERVED_RANGE:
            return

        try:
            file_ = xnat_object.resource("DICOM").files()[0].get()
            dicom_data = dicom.read_file(file_)
            contrast = extract_contrast(dicom_data)
            os.remove(file_)
        except StopIteration:
            contrast = "unknown"
        xnat_object.attrs.set("type", contrast)

        if verbose:
            print("updated type to {}".format(contrast))

        time.sleep(wait)

    except Exception as error:

        if raise_:
            raise error
        else:
            print("Error updating scan type for object {}:".format(xnat_object))
            print(error.__repr__())


def process(xnat_object, wait=1, verbose=True, raise_=False):

    if not hasattr(xnat_object, "__iter__"):

        if not xnat_object.exists():

            error_string = "Object {} does not exist.".format(xnat_object)
            if raise_:
                raise KeyError(error_string)
            else:
                return

        elif type(xnat_object) == res.Project:

            process(xnat_object.subjects(), wait, verbose, raise_)

        elif type(xnat_object) == res.Subject:

            process(xnat_object.experiments(), wait, verbose, raise_)

        elif type(xnat_object) == res.Experiment:

            process(xnat_object.scans(), wait, verbose, raise_)

        elif type(xnat_object) == res.Scan:

            process_single(xnat_object, wait, verbose, raise_)

        else:

            error_string = "Cannot update scan type for object of type {}".format(
                type(xnat_object)
            )
            if raise_:
                raise TypeError(error_string)
            else:
                print(error_string)
                return

    else:

        if len(list(xnat_object)) == 0:
            error_string = "Empty collection of objects"
            if raise_:
                raise IOError(error_string)
            else:
                print(error_string)

        for o in xnat_object:

            process(o, wait, verbose, raise_)


# =============================================================================
# MAIN METHOD
# =============================================================================


def main():

    parser = argparse.ArgumentParser(
        description="Update correct contrast for all scans at XNAT endpoint"
    )
    parser.add_argument("-url", type=str, help="URL pointing to an XNAT instance")
    parser.add_argument("-s", "--select", type=str, help="API endpoint")
    parser.add_argument("-u", "--user", type=str, help="XNAT user")
    parser.add_argument("-p", "--password", type=str, help="XNAT user password")
    parser.add_argument(
        "-w", "--wait", type=int, default=1, help="Seconds to wait after PUT"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Toggle verbose output"
    )
    parser.add_argument(
        "-r",
        "--raiseerrors",
        action="store_true",
        help="Raise errors if update not possible",
    )
    args = parser.parse_args()

    if args.verbose:
        print("Running {}".format(__file__))
        print(args)

    url = args.url
    if url.endswith("/"):
        url = url[:-1]
    select = args.select
    if not select.startswith("/"):
        select = "/" + select
    if select.startswith("/archive"):
        select = select[8:]
    if select.startswith("/data"):
        select = select[5:]

    interface = pyxnat.Interface(url, args.user, args.password)
    endpoint = interface.select(select)

    process(endpoint, args.wait, args.verbose, args.raiseerrors)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
