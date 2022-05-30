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

Create segmentations for all experiments under a given XNAT endpoint

REQUIRES
--------

lasagne
numpy
pyxnat
scikit-image
SimpleITK
theano

TODO
----



"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

import argparse
import pickle
import csv
from datetime import datetime
import imp
import os
import shutil
import time
import traceback
import subprocess as subp

import nibabel as nib
import numpy as np
import pyxnat
import pyxnat.core.resources as res
import torch

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

CONTRASTS = ("T1", "T1ce", "T2", "FLAIR", "ADC", "CBV")
RESERVED_RANGE = ("8999", "9990", "9991", "9992", "9993", "9994", "9995")

# =============================================================================
# METHODS & CLASSES
# =============================================================================


def process_single(xnat_object, wait=1, verbose=True, raise_=False):

    try:

        # create download folder
        download_dir = os.path.join("tmp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        download_dir = os.path.abspath(download_dir)
        os.makedirs(download_dir)

        # download
        files = {}
        for scan in xnat_object.scans():
            scan_number = scan.attrs.get("ID")
            if scan_number in RESERVED_RANGE:
                continue
            scan_type = scan.attrs.get("type")
            if scan_type in CONTRASTS:
                if scan_type == "T1":
                    file_name = "T1_r2s_bet.nii.gz"
                else:
                    file_name = "{}_r2s_bet_regT1.nii.gz".format(scan_type)
                xnat_file = scan.resource("NIFTI").file(file_name)
                if not xnat_file.exists():
                    continue
                xnat_file.get(os.path.join(download_dir, file_name))
                files[scan_type] = os.path.join(download_dir, file_name)
                if verbose:
                    print("Downloaded {} from scan {}".format(file_name, scan))

        # check if we have all four contrasts (ignore ADC and CBV)
        for contrast in CONTRASTS[:4]:
            if contrast not in files.keys():
                raise IOError("Could not find file for contrast {}.".format(contrast))

        # prepare Decathlon format for nnU-Net and run segmentation
        nnunet_data_dir = os.path.join(download_dir, "nnunet")
        os.mkdir(nnunet_data_dir)
        for c, contrast in enumerate(("T1", "T1ce", "T2", "FLAIR")):
            shutil.move(
                files[contrast],
                os.path.join(nnunet_data_dir, "segmentation_{:04d}.nii.gz".format(c)),
            )
        previous_wd = os.getcwd()
        os.chdir("/scripts/nnUnet/nnunet")
        output = subp.check_output(
            [
                "python3",
                "inference/predict_simple.py",
                "-i",
                nnunet_data_dir,
                "-o",
                download_dir,
                "-t",
                "Task12_BrainTumorIntern",
            ]
        )
        os.chdir(previous_wd)
        for c, contrast in enumerate(("T1", "T1ce", "T2", "FLAIR")):
            shutil.move(
                os.path.join(nnunet_data_dir, "segmentation_{:04d}.nii.gz".format(c)),
                files[contrast],
            )
        if verbose:
            print(output)

        # FAST for CBV
        if "CBV" in files:

            wm_seg_file = files["T1"].replace(".nii.gz", "_seg_2.nii.gz")
            cmd = ["fast", "-v", "-g", files["T1"]]
            output = subp.check_output(cmd)
            if verbose:
                print(output)

        # get volumes and save as csv
        seg = nib.load(os.path.join(download_dir, "segmentation.nii.gz")).get_fdata()
        volume_labels = [
            "Edema / Nonenhancing Tumor",
            "Enhancing Tumor",
            "Necrosis",
            "Whole Tumor",
        ]
        volumes = []
        volumes.append(np.sum(seg == 1) / 1000.0)
        volumes.append(np.sum(seg == 2) / 1000.0)
        volumes.append(np.sum(seg == 3) / 1000.0)
        volumes.append(np.sum(volumes))
        if "ADC" in files:
            adc_data = nib.load(files["ADC"]).get_fdata()
            median_adc = []
            median_adc.append(np.median(adc_data[seg == 1]))
            median_adc.append(np.median(adc_data[seg == 2]))
            median_adc.append(np.median(adc_data[seg == 3]))
            median_adc.append(np.median(adc_data[seg != 0]))
        if "CBV" in files:
            cbv_data = nib.load(files["CBV"]).get_fdata()
            wm_seg = nib.load(wm_seg_file).get_fdata().astype(np.bool)
            wm_seg[seg != 0] = 0
            std_cbv_wm_healthy = np.std(cbv_data[wm_seg])
            median_cbv = []
            median_cbv.append(np.median(cbv_data[seg == 1]) / std_cbv_wm_healthy)
            median_cbv.append(np.median(cbv_data[seg == 2]) / std_cbv_wm_healthy)
            median_cbv.append(np.median(cbv_data[seg == 3]) / std_cbv_wm_healthy)
            median_cbv.append(np.median(cbv_data[seg != 0]) / std_cbv_wm_healthy)
        with open(os.path.join(download_dir, "volumes.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(volume_labels)
            writer.writerow(volumes)
            if "ADC" in files:
                writer.writerow(median_adc)
            if "CBV" in files:
                writer.writerow(median_cbv)
        if verbose:
            print("Calculated volumes")
            print(volume_labels)
            print(volumes)
            if "ADC" in files:
                print(median_adc)
            if "CBV" in files:
                print(median_cbv)

        # upload
        seg_resource = xnat_object.scan("9990").resource("NIFTI")
        if not seg_resource.exists():
            seg_resource.create()
            seg_resource.parent().attrs.set("type", "Segmentation")
        seg_resource.file("segmentation.nii.gz").put(
            os.path.join(download_dir, "segmentation.nii.gz"), overwrite=True
        )
        xnat_object.scan("9990").resource("CSV").file("volumes.csv").put(
            os.path.join(download_dir, "volumes.csv"), overwrite=True
        )
        if "CBV" in files:
            seg_resource.file(os.path.basename(wm_seg_file)).put(
                wm_seg_file, overwrite=True
            )
        if verbose:
            print("Uploaded segmentation for experiment {}".format(xnat_object))

        time.sleep(wait)

    except Exception as e:

        if raise_:
            raise e
        else:
            print("Couldn't create segmentation for experiment {}".format(xnat_object))
            print(traceback.format_exc())
            print(e.__repr__())

    finally:

        # clean up
        shutil.rmtree(download_dir)
        if verbose:
            print("Removed all temporary data")


def process(xnat_object, wait=1, verbose=True, raise_=False, **kwargs):

    if not hasattr(xnat_object, "__iter__"):

        if not xnat_object.exists():

            error_string = "Object {} does not exist.".format(xnat_object)
            if raise_:
                raise KeyError(error_string)
            else:
                return

        elif type(xnat_object) == res.Project:

            process(xnat_object.subjects(), wait, verbose, raise_, **kwargs)

        elif type(xnat_object) == res.Subject:

            process(xnat_object.experiments(), wait, verbose, raise_, **kwargs)

        elif type(xnat_object) == res.Experiment:

            process_single(xnat_object, wait, verbose, raise_, **kwargs)

        elif type(xnat_object) == res.Scan:

            process(xnat_object.parent(), wait, verbose, raise_, **kwargs)

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

            process(o, wait, verbose, raise_, **kwargs)


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
