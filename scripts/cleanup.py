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

Preprocess all experiments under a given XNAT endpoint

REQUIRES
--------

pyxnat
FSL command line tools

TODO
----



"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

import argparse
import os
import shutil
import subprocess as subp
import time
import traceback
import multiprocessing as mp
from functools import partial

import nibabel as nib
import numpy as np
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

CONTRASTS = ("T1", "T1ce", "T2", "FLAIR", "ADC", "CBV")
DELETE = ("T1sub", )
RESERVED_RANGE = ("8999", "9990", "9991", "9992", "9993", "9994", "9995")

# =============================================================================
# METHODS & CLASSES
# =============================================================================


def process_single(xnat_object, wait=1, verbose=True, raise_=False):

    try:

        base_scans = {c: [] for c in CONTRASTS}
        delete = []

        for scan in xnat_object.scans():

            scan_number = scan.attrs.get("ID")
            if scan_number in RESERVED_RANGE:
                continue

            scan_type = scan.attrs.get("type")
            if scan_type in base_scans:
                base_scans[scan_type].append(scan)
            if scan_type == "FLAIR3D":
                base_scans["FLAIR"].append(scan)
            if scan_type in DELETE:
                delete.append(scan)

        for contrast, scans in base_scans.items():

            if contrast == "FLAIR":

                # if we only have 3d flairs, we use those by setting the types to regular flair,
                # otherwise we remove the 3d ones from the option pool
                scans_regular = []
                scans_3d = []
                for scan in scans:
                    type_ = scan.attrs.get("type")
                    if type_ == "FLAIR":
                        scans_regular.append(scan)
                    elif type_ == "FLAIR3D":
                        scans_3d.append(scan)
                if len(scans_regular) > 0:
                    scans = scans_regular
                else:
                    for scan in scans_3d:
                        scan.attrs.set("type", "FLAIR")
                    scans = scans_3d

            if len(scans) <= 1:
                continue

            ids = []
            for scan in scans:
                try:
                    ids.append(int(scan.attrs.get("ID")))
                except Exception as e:
                    print("Couldn't translate ID into int for scan {}".format(scan))
                    ids.append(1e9)
            selected_index = int(np.argmin(ids))
            for s in range(len(scans)):
                if s != selected_index:
                    delete.append(scans[s])

        for scan in delete:
            id_ = scan.attrs.get("ID")
            type_ = scan.attrs.get("type")
            scan.delete()
            if verbose: print("Deleted scan {} with type {}".format(id_, type_))

        time.sleep(wait)

    except Exception as e:

        if raise_:
            raise e
        else:
            print("Couldn't clean up files for experiment {}".format(xnat_object))
            print(e.__repr__())


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

            process_single(xnat_object, wait, verbose, raise_)

        elif type(xnat_object) == res.Scan:

            process(xnat_object.parent(), wait, verbose, raise_)

        else:

            error_string = "Cannot process object of type {}".format(type(xnat_object))
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
        description="Preprocess all experiments at an XNAT endpoint")
    parser.add_argument("-url", type=str,
                        help="URL pointing to an XNAT instance")
    parser.add_argument("-s", "--select", type=str,
                        help="API endpoint")
    parser.add_argument("-u", "--user", type=str,
                        help="XNAT user")
    parser.add_argument("-p", "--password", type=str,
                        help="XNAT user password")
    parser.add_argument("-w", "--wait", type=int, default=1,
                        help="Seconds to wait after PUT")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Toggle verbose output")
    parser.add_argument("-r", "--raiseerrors", action="store_true",
                        help="Raise errors if update not possible")
    args = parser.parse_args()

    if args.verbose:
        print("Running {}".format(__file__))
        print(args)

    url = args.url
    if url.endswith("/"): url = url[:-1]
    select = args.select
    if not select.startswith("/"): select = "/" + select
    if select.startswith("/archive"): select = select[8:]
    if select.startswith("/data"): select = select[5:]

    interface = pyxnat.Interface(url, args.user, args.password)
    endpoint = interface.select(select)

    process(endpoint, args.wait, args.verbose, args.raiseerrors)

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
# -*- coding: utf-8 -*-
