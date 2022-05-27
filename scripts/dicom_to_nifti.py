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

Convert all scans under a given XNAT endpoint to NIFTI

REQUIRES
--------

pyxnat
dcm2nii command line tool (from mricron)

TODO
----



"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

import argparse
from datetime import datetime
import glob
import os
import shutil
import subprocess as subp
import time
import traceback
import multiprocessing as mp
from functools import partial

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
RESERVED_RANGE = ("8999", "9990", "9991", "9992", "9993", "9994", "9995")

# =============================================================================
# METHODS & CLASSES
# =============================================================================


def convert(input_, download_dir):
    try:
        location, output_name = input_
        cmd = ["mcverter", "-o", download_dir, "-f", "nifti", "-v", "-n",
            "-F", output_name, location]
        output = subp.check_output(cmd)
    except Exception as e:
        output = e.__repr__()
    return output


def gzip(file_):
    try:
        return subp.check_output(["gzip", file_])
    except Exception as e:
        return e.__repr__()


def process_single(xnat_object, wait=1, verbose=True, raise_=False):

    try:

        # create download folder
        download_dir = os.path.join("tmp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(download_dir)

        # get scans
        scans = []
        for scan in xnat_object.scans():
            id_ = scan.attrs.get("ID")
            type_ = scan.attrs.get("type")
            if id_ in RESERVED_RANGE or type_ not in CONTRASTS:
                continue
            else:
                files = scan.resource("DICOM").files()
                if len(list(files)) == 0:
                    info_str = "Couldn't find any DICOM files for scan {}".format(scan)
                    if raise_:
                        raise RuntimeError(info_str)
                    else:
                        print(info_str)
                        continue
                os.mkdir(os.path.join(download_dir, id_))
                list(map(lambda x: x.get(os.path.join(download_dir, id_, x.id())), files))
                scans.append(scan)
                if verbose: print("Downloaded DICOM data for scan {}".format(scan))

        # convert to NIfTI
        locations = []
        for scan in scans:
            locations.append(os.path.join(download_dir, scan.attrs.get("ID")))
        names = [scan.attrs.get("type") for scan in scans]
        p = mp.Pool(min(len(scans), mp.cpu_count()))
        outputs = p.map(partial(convert, download_dir=download_dir), zip(locations, names))
        p.close()
        p.join()
        if verbose:
            for output in outputs:
                print(output)

        # zip
        nifti_files = glob.glob(os.path.join(download_dir, "*.nii"))
        p = mp.Pool(min(len(nifti_files), mp.cpu_count()))
        outputs = p.map(gzip, nifti_files)
        p.close()
        p.join()
        if verbose:
            for output in outputs:
                print(output)
        nifti_files = list(map(lambda x: x + ".gz", nifti_files))

        # upload
        for scan in scans:
            resource = scan.resource("NIFTI")
            if not resource.exists(): resource.create()
            for file_ in nifti_files:
                if os.path.basename(file_).split(".")[0] == scan.attrs.get("type"):
                    resource.file(os.path.basename(file_)).put(file_, overwrite=True)
                    if verbose: print("Uploaded {}".format(os.path.basename(file_)))
                    time.sleep(wait)

    except Exception as e:

        if raise_:
            raise e
        else:
            print("Couldn't create NIFTI files for scan {}".format(xnat_object))
            print(e.__repr__())

    finally:

        # clean up
        shutil.rmtree(download_dir)
        if verbose: print("Removed all temporary data")


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

            error_string = "Cannot convert object of type {}".format(type(xnat_object))
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
        description="Convert all scans at XNAT endpoint to NIFTI")
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
