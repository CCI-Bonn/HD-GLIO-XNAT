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
from datetime import datetime
import os
import re
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

# Edited by Hagen Meredig Wed May 25 2022

CONTRASTS = ("T1", "T1ce", "T2", "FLAIR")
KEEP_AND_IGNORE = ("ADC", "CBV", "SWI")
RESERVED_RANGE = ("8999", "9990", "9991", "9992", "9993", "9994", "9995")

# =============================================================================
# METHODS & CLASSES
# =============================================================================


def reorient2std(file_):
    outfile = file_.replace(".nii.gz", "_r2s.nii.gz")
    cmd = ["fslreorient2std", file_, outfile]
    output = subp.check_output(cmd)
    return outfile, output


def register_to_t1(file_, reference_file):
    outfile = file_.replace(".nii.gz", "_regT1.nii.gz")
    cmd = [
        "flirt",
        "-in",
        file_,
        "-ref",
        reference_file,
        "-out",
        outfile,
        "-dof",
        "6",
        "-interp",
        "spline",
    ]
    output = subp.check_output(cmd)
    return outfile, output


def process_single(xnat_object, wait=1, verbose=True, raise_=False):

    try:

        # create download folder
        download_dir = os.path.join("tmp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(download_dir)

        # download the scans we need and remove the rest (REMOVING IS NOW DONE BY SEPARATE SCRIPT!!)
        scans = {}
        unused = []
        already_kept_and_ignored = []
        for scan in xnat_object.scans():
            scan_number = scan.attrs.get("ID")
            if scan_number in RESERVED_RANGE:
                continue
            scan_type = scan.attrs.get("type")
            if (
                scan_type in KEEP_AND_IGNORE
                and scan_type not in already_kept_and_ignored
            ):
                scans[scan_type] = scan
                already_kept_and_ignored.append(scan_type)
                continue
            file_name = scan_type + ".nii.gz"
            if scan_type in CONTRASTS:
                xnat_file = scan.resource("NIFTI").file(file_name)
                if not xnat_file.exists():
                    unused.append(scan)
                    continue
                xnat_file.get(os.path.join(download_dir, file_name))
                scans[scan_type] = scan
                if verbose:
                    print("Downloaded {} from scan {}".format(file_name, scan))
            else:
                unused.append(scan)
        # for scan in unused:
        #     scan.delete()
        #     if verbose: print("Deleted scan {}".format(scan))

        # check if we have all contrasts
        for contrast in CONTRASTS:
            if not os.path.exists(os.path.join(download_dir, contrast + ".nii.gz")):
                raise IOError("Could not find file for contrast {}.".format(contrast))

        files = [
            os.path.join(download_dir, "T1.nii.gz"),
            os.path.join(download_dir, "T1ce.nii.gz"),
            os.path.join(download_dir, "T2.nii.gz"),
            os.path.join(download_dir, "FLAIR.nii.gz"),
        ]

        # Reorient to standard
        p = mp.Pool(min(len(files), mp.cpu_count()))
        results = p.map(reorient2std, files)
        p.close()
        p.join()
        files, outputs = list(zip(*results))
        files = list(files)
        if verbose:
            for output in outputs:
                print(output)

        # # Brain extraction (do not parallelize because we either use gpu or too much memory)
        # bet_params = os.path.join(os.path.dirname(__file__), "bet", "model_final.model")
        # bet_config = os.path.join(os.path.dirname(__file__), "bet", "basic_config_just_like_braintumor.py")
        # mask_files = []
        # for f, file_ in enumerate(files):
        #     mask_file = file_.replace(".nii.gz", "_mask.nii.gz")
        #     run_bet(file_, mask_file, bet_params, bet_config, device, True)
        #     if verbose: print("Created brain mask for {}".format(file_))
        #     mask_files.append(mask_file)
        #     outfile = file_.replace(".nii.gz", "_bet.nii.gz")
        #     cmd = ["fslmaths", file_, "-mas", mask_file, outfile]
        #     output = subp.check_output(cmd)
        #     if verbose: print(output)
        #     files[f] = outfile

        # Brain extraction (do not parallelize because we run on gpu)
        mask_files = []
        for f, file_ in enumerate(files):
            output = subp.check_output(["hd-bet", "-i", file_, "-device", "0"])
            if verbose:
                print(output)
            mask_file = file_.replace(".nii.gz", "_bet_mask.nii.gz")
            file_ = file_.replace(".nii.gz", "_bet.nii.gz")
            mask_files.append(mask_file)
            cmd = ["fslmaths", file_, "-mas", mask_file, file_]
            output = subp.check_output(cmd)
            files[f] = file_
            if verbose:
                print(output)

        # Register to T1
        p = mp.Pool(min(len(files) - 1, mp.cpu_count()))
        results = p.map(partial(register_to_t1, reference_file=files[0]), files[1:])
        p.close()
        p.join()
        new_files, outputs = list(zip(*results))
        files[1:] = new_files
        if verbose:
            for output in outputs:
                print(output)

        # Make T1 sub
        t1_file = files[0]
        t1ce_file = files[1]
        sub_file = files[0].replace("T1", "T1sub")
        for f in (t1_file, t1ce_file):
            nifti = nib.load(f)
            data = nifti.get_data()
            data = data - np.mean(data)
            data = data / np.std(data)
            new = nib.Nifti1Image(data, nifti.affine, nifti.header)
            outname = f.replace(".nii.gz", "_norm.nii.gz")
            nib.save(new, outname)
        cmd = [
            "fslmaths",
            t1ce_file.replace(".nii.gz", "_norm.nii.gz"),
            "-sub",
            t1_file.replace(".nii.gz", "_norm.nii.gz"),
            sub_file,
        ]
        output = subp.check_output(cmd)
        if verbose:
            print(output)

        # Process ADC separately
        if "ADC" in scans:

            adc_file = "ADC.nii.gz"
            xnat_file = scans["ADC"].resource("NIFTI").file(adc_file)
            xnat_file.get(os.path.join(download_dir, adc_file))
            adc_file = os.path.join(download_dir, adc_file)

            outfile = adc_file.replace(".nii.gz", "_r2s.nii.gz")
            cmd = ["fslreorient2std", adc_file, outfile]
            output = subp.check_output(cmd)
            if verbose:
                print(output)
            adc_file = outfile

            mask_file = adc_file.replace(".nii.gz", "_mask.nii.gz")
            cmd = [
                "flirt",
                "-in",
                mask_files[0],
                "-ref",
                adc_file,
                "-out",
                mask_file,
                "-dof",
                "6",
                "-interp",
                "nearestneighbour",
            ]
            output = subp.check_output(cmd)
            if verbose:
                print(output)

            outfile = adc_file.replace(".nii.gz", "_bet.nii.gz")
            cmd = ["fslmaths", adc_file, "-mas", mask_file, outfile]
            output = subp.check_output(cmd)
            if verbose:
                print(output)
            adc_file = outfile

            outfile = adc_file.replace(".nii.gz", "_regT1.nii.gz")
            cmd = [
                "flirt",
                "-in",
                adc_file,
                "-ref",
                files[0],
                "-out",
                outfile,
                "-dof",
                "6",
                "-interp",
                "spline",
            ]
            output = subp.check_output(cmd)
            if verbose:
                print(output)
            adc_file = outfile

            files.append(adc_file)

        # Process CBV separately
        if "CBV" in scans:

            cbv_file = "CBV.nii.gz"
            xnat_file = scans["CBV"].resource("NIFTI").file(cbv_file)
            xnat_file.get(os.path.join(download_dir, cbv_file))
            cbv_file = os.path.join(download_dir, cbv_file)

            outfile = cbv_file.replace(".nii.gz", "_r2s.nii.gz")
            cmd = ["fslreorient2std", cbv_file, outfile]
            output = subp.check_output(cmd)
            if verbose:
                print(output)
            cbv_file = outfile

            mask_file = cbv_file.replace(".nii.gz", "_mask.nii.gz")
            cmd = [
                "flirt",
                "-in",
                mask_files[0],
                "-ref",
                cbv_file,
                "-out",
                mask_file,
                "-dof",
                "6",
                "-interp",
                "nearestneighbour",
            ]
            output = subp.check_output(cmd)
            if verbose:
                print(output)

            outfile = cbv_file.replace(".nii.gz", "_bet.nii.gz")
            cmd = ["fslmaths", cbv_file, "-mas", mask_file, outfile]
            output = subp.check_output(cmd)
            if verbose:
                print(output)
            cbv_file = outfile

            outfile = cbv_file.replace(".nii.gz", "_regT1.nii.gz")
            cmd = [
                "flirt",
                "-in",
                cbv_file,
                "-ref",
                files[0],
                "-out",
                outfile,
                "-dof",
                "6",
                "-interp",
                "spline",
            ]
            output = subp.check_output(cmd)
            if verbose:
                print(output)
            cbv_file = outfile

            files.append(cbv_file)

        # upload
        for f, file_ in enumerate(files):
            file_name = os.path.basename(file_)
            scan = scans[file_name.split("_")[0]]
            scan.resource("NIFTI").file(file_name).put(file_, overwrite=True)
            if verbose:
                print("Uploaded {}".format(file_name))
        seg_resource = xnat_object.scan("9990").resource("NIFTI")
        if not seg_resource.exists():
            seg_resource.create()
            seg_resource.parent().attrs.set("type", "Segmentation")
        for mask_file in mask_files:
            seg_resource.file(os.path.basename(mask_file)).put(
                mask_file, overwrite=True
            )
        sub_resource = xnat_object.scan("8999").resource("NIFTI")
        if not sub_resource.exists():
            sub_resource.create()
            sub_resource.parent().attrs.set("type", "T1sub")
        sub_resource.file(os.path.basename(sub_file)).put(sub_file, overwrite=True)

        time.sleep(wait)

    except Exception as e:

        if raise_:
            raise e
        else:
            print("Couldn't preprocess files for experiment {}".format(xnat_object))
            print(e.__repr__())

    finally:

        # clean up
        shutil.rmtree(download_dir)
        if verbose:
            print("Removed all temporary data")


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
        description="Preprocess all experiments at an XNAT endpoint"
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
    parser.add_argument("-d", "--device", type=str, default="0")
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    process(endpoint, args.wait, args.verbose, args.raiseerrors)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
# -*- coding: utf-8 -*-
