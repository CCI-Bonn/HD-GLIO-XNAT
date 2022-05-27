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

Create segmentation DICOMs for XNAT experiment

REQUIRES
--------

matplotlib
numpy
pyxnat
pdf2dcm command line tool (from DCMTK)

TODO
----



"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

import argparse
import copy
from datetime import datetime
import os
import re
import shutil
import subprocess as subp
import time
import traceback
import multiprocessing as mp
from functools import partial

import pydicom as dicom
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import nibabel as nib
import numpy as np
import pyxnat
import pyxnat.core.resources as res
from skimage import color
from skimage.transform import resize

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

CONTRASTS = ("T1ce", "FLAIR", "T1sub")  # T1 is hardcoded as reference
CLASSES = ("Edema / Nonenhancing Tumor", "Enhancing Tumor")
COLORS = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))

# =============================================================================
# METHODS & CLASSES
# =============================================================================


def create_random_string(length, elements="0123456789"):

    result = ""
    while len(result) < length:
        result += elements[np.random.randint(len(elements))]
    return result


def make_random_uids(sopinstanceuid_old, seriesinstanceuid_old):

    np.random.seed(int(str(time.time()).replace(".", "")) % (2**32 - 1))

    sop_split = sopinstanceuid_old.split(".")
    series_split = seriesinstanceuid_old.split(".")

    sop_shape = list(map(len, sop_split))
    series_shape = list(map(len, series_split))

    # find how many entries match
    match = 0
    max_match = min(len(sop_shape), len(series_shape)) - 1
    while match < max_match and sop_split[match] == series_split[match]:
        match += 1

    # we will create random numbers for nonmatching entries
    # need at least 8 random numbers
    sop_num_rand = sum(sop_shape[match:])
    series_num_rand = sum(series_shape[match:])
    while sop_num_rand < 8 or series_num_rand < 8:
        match -= 1
        sop_num_rand = sum(sop_shape[match:])
        series_num_rand = sum(series_shape[match:])

    sopinstanceuid_new = sop_split[:match]
    seriesinstanceuid_new = series_split[:match]
    for num in sop_shape[match:]:
        random_string = "0"
        while random_string.startswith("0"):
            random_string = create_random_string(num)
        sopinstanceuid_new.append(random_string)
    for num in series_shape[match:]:
        random_string = "0"
        while random_string.startswith("0"):
            random_string = create_random_string(num)
        seriesinstanceuid_new.append(random_string)

    sopinstanceuid_new = ".".join(sopinstanceuid_new)[:64]
    seriesinstanceuid_new = ".".join(seriesinstanceuid_new)[:64]

    return sopinstanceuid_new, seriesinstanceuid_new


def step_uid(uid_old):

    remaining_space = 64 - len(uid_old)
    uid_split = uid_old.split(".")

    if remaining_space <= 0:

        for i in range(1, len(uid_split) + 1):
            old_entry = uid_split[-i]
            new_entry = str(int(old_entry) + 1)
            if not len(new_entry) > len(old_entry):
                uid_split[-i] = new_entry
                break
            else:
                uid_split[-i] = new_entry[1:]

    else:

        uid_split[-1] = str(int(uid_split[-1]) + 1)

    return ".".join(uid_split)


def adjust_shape(original_shape, n_data):

    original_shape = list(original_shape)
    n_pixels = original_shape[0] * original_shape[1] * 3
    next_index_to_change = 1

    while n_pixels != n_data:

        if n_pixels < n_data:
            original_shape[next_index_to_change] += 1
        else:
            original_shape[next_index_to_change] -= 1
            next_index_to_change = 1 - next_index_to_change
            n_pixels = original_shape[0] * original_shape[1] * 3

    return tuple(original_shape)


def make_warning_image(image_size):

    image_size = np.array(image_size, dtype=np.int32)

    dpi = 100
    size_in = image_size / float(dpi)

    fig = Figure((size_in[1], size_in[0]), dpi)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    disclaimer = "NOT CLINICALLY VALIDATED\nDO NOT REPRODUCE OR DISTRIBUTE"
    ax.text(
        0.5,
        0,
        disclaimer,
        fontsize=image_size[1] // (30 * dpi / 100),
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    ax.axis("off")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).astype(np.float64)
    target_shape = adjust_shape(image_size, image.shape[0])
    image = image.reshape(list(target_shape) + [3])[..., 0]

    image = resize(
        image, image_size, order=3, mode="constant", anti_aliasing=False
    ).round()
    image[image > 255] = 255
    image = image.astype(np.uint8)

    return 255 - image


def delete_tag(dcm_data, x, y):

    try:
        del dcm_data[x, y]
    except KeyError:
        pass


def make_overlay(image, overlay, alpha=0.3):

    image_hsv = color.rgb2hsv(image)
    overlay_hsv = color.rgb2hsv(overlay)

    image_hsv[..., 0] = overlay_hsv[..., 0]
    image_hsv[..., 1] = alpha * overlay_hsv[..., 1]
    image = color.hsv2rgb(image_hsv)
    image = (image * 255).round().astype(np.uint8)

    return image


def dcm_attr_fallback(dcm_data, attr, control=None, default=None):

    if control is not None and default is None:
        raise ValueError("control requires default")

    try:
        obj = getattr(dcm_data, attr)
        if control is None:
            if hasattr(obj, "encodings") and len(getattr(obj, "encodings")) > 0:
                result = obj.encode(getattr(obj, "encodings")[0])
            else:
                result = obj.encode("latin-1")
        else:
            result = str(obj)
    except AttributeError:
        result = ""

    if control is None:
        if result == "" and default is not None:
            result = default

    else:
        if isinstance(control, str):
            if not re.match(control, result):
                result = default
        elif hasattr(control, "__iter__"):
            if result not in control:
                result = default
        else:
            raise TypeError("control must be regex or iterable")

    return result


def resample(input_):

    file_, contrast = input_
    if contrast == "Segmentation":
        cmd = [
            "flirt",
            "-in",
            file_,
            "-ref",
            file_,
            "-out",
            file_,
            "-interp",
            "nearestneighbour",
            "-applyisoxfm",
            "0.8",
        ]
    else:
        cmd = [
            "flirt",
            "-in",
            file_,
            "-ref",
            file_,
            "-out",
            file_,
            "-interp",
            "spline",
            "-applyisoxfm",
            "0.8",
        ]
    return subp.check_output(cmd)


def get_min_max(file_):

    data = nib.load(file_).get_data()
    return np.min(data), np.max(data)


def nifti2dicom(input_, reference_dcm_data, download_dir):

    file_, contrast, c = input_

    os.mkdir(os.path.join(download_dir, contrast))
    cmd = [
        "nifti2dicom",
        "-i",
        file_,
        "-o",
        os.path.join(download_dir, contrast),
        "-y",
        "--sopclassuid",
        "1.2.840.10008.5.1.4.1.1.7",
        "--modality",
        "MR",
        "--seriesdescription",
        "{} Segmentation".format(contrast),
        "--studydescription",
        dcm_attr_fallback(reference_dcm_data, "StudyDescription"),
        "--manufacturer",
        "nifti2dicom",
        "--institutionname",
        dcm_attr_fallback(
            reference_dcm_data,
            "InstitutionName",
            default="Dept. of Neuroradiology, Heidelberg University Hospital",
        ),
        "--studydate",
        dcm_attr_fallback(reference_dcm_data, "StudyDate", "\d{8}", "18990101"),
        "--patientname",
        dcm_attr_fallback(reference_dcm_data, "PatientName"),
        "--patientid",
        dcm_attr_fallback(reference_dcm_data, "PatientID"),
        "--patientdob",
        dcm_attr_fallback(reference_dcm_data, "PatientBirthDate", "\d{8}", "18990101"),
        "--patientsex",
        dcm_attr_fallback(
            reference_dcm_data, "PatientSex", ["M", "m", "F", "f", "O", "o"], "O"
        ),
        "--patientweight",
        dcm_attr_fallback(reference_dcm_data, "PatientWeight"),
        "--patientage",
        dcm_attr_fallback(reference_dcm_data, "PatientAge"),
        "--studyinstanceuid",
        dcm_attr_fallback(reference_dcm_data, "StudyInstanceUID"),
        "--studyid",
        dcm_attr_fallback(reference_dcm_data, "StudyID"),
        "--seriesdate",
        dcm_attr_fallback(reference_dcm_data, "SeriesDate", "\d{8}", "18990101"),
        "--acquisitiondate",
        dcm_attr_fallback(reference_dcm_data, "AcquisitionDate", "\d{8}", "18990101"),
        "--seriesnumber",
        "999" + str(c),
        # "--seriesinstanceuid", "999" + str(c),
        "--studytime",
        dcm_attr_fallback(reference_dcm_data, "StudyTime"),
        "--seriestime",
        dcm_attr_fallback(reference_dcm_data, "SeriesTime"),
        "--acquisitiontime",
        dcm_attr_fallback(reference_dcm_data, "AcquisitionTime"),
    ]
    return subp.check_output(cmd)


def make_overlay_dicoms(input_, download_dir, seg_data):

    contrast, min_max = input_

    dicom_dir = os.path.join(download_dir, contrast)
    output_dir = os.path.join(download_dir, contrast + "_overlay")
    os.mkdir(output_dir)

    warning_image = None

    for df, dicom_file in enumerate(sorted(os.listdir(dicom_dir))):

        current_file_new = dicom.read_file(os.path.join(dicom_dir, dicom_file))

        if df == 0:
            warning_image = make_warning_image(current_file_new.pixel_array.shape)

        # skip empty slices
        if np.all(current_file_new.pixel_array == 0):
            continue

        current_data = np.dstack(
            (
                current_file_new.pixel_array,
                current_file_new.pixel_array,
                current_file_new.pixel_array,
            )
        ).astype(np.float64)
        current_data = (current_data - min_max[0]) / (min_max[1] - min_max[0])

        current_data = make_overlay(current_data, seg_data[df])
        current_data = current_data.transpose(2, 0, 1)

        for i in range(current_data.shape[0]):
            current_data[i] += warning_image.astype(bool) * (
                warning_image - current_data[i]
            )

        # make number of entries even so we don't need trailing null
        if np.product(current_data.shape) % 2 != 0:
            new_shape = list(current_data.shape)
            new_shape[1] += 1
            new_current_data = np.zeros(new_shape, dtype=current_data.dtype)
            new_current_data[:, :-1, :] = current_data
            current_data = new_current_data

        # try:
        #     print(current_file_new.pixel_array.shape, current_data.shape)
        #     current_file_new.pixel_array = current_data
        # except:
        #     import IPython
        #     IPython.embed()
        current_file_new.PixelData = current_data.tostring()
        current_file_new.Rows, current_file_new.Columns = current_data.shape[1:]
        current_file_new.BitsAllocated = 8
        current_file_new.BitsStored = 8
        current_file_new.HighBit = 7
        current_file_new.SamplesPerPixel = 3
        current_file_new.PhotometricInterpretation = "RGB"
        current_file_new.PlanarConfiguration = 1
        current_file_new.PixelRepresentation = 0
        try:
            current_file_new.PixelSpacing = current_file_new.NominalScannedPixelSpacing
        except Exception as e:
            pass

        current_file_new.save_as(os.path.join(output_dir, dicom_file))


def make_preview_images(contrast, download_dir):

    data = []
    for f in sorted(os.listdir(os.path.join(download_dir, contrast + "_overlay"))):
        data.append(
            dicom.read_file(
                os.path.join(download_dir, contrast + "_overlay", f)
            ).pixel_array
        )

    n_images = len(data)
    n_images_y = int(np.sqrt(n_images))
    n_images_x = n_images // n_images_y
    if n_images % n_images_y != 0:
        n_images_x += 1

    image = np.zeros(
        (n_images_x * data[0].shape[0], n_images_y * data[0].shape[1], 3),
        dtype=data[0].dtype,
    )

    for i in range(n_images):
        x = i // n_images_y
        y = i % n_images_y
        image[
            x * data[0].shape[0] : (x + 1) * data[0].shape[0],
            y * data[0].shape[1] : (y + 1) * data[0].shape[1],
            :,
        ] = data[i]

    image_name = "{}_Segmentation.jpg".format(contrast)
    thumbnail_name = image_name.replace(".jpg", "_t.jpg")

    aspect = float(image.shape[0]) / float(image.shape[1])
    fig = plt.figure(figsize=(8, 8 * aspect))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(image, extent=(0, 1, 1, 0))
    ax.axis("tight")
    ax.axis("off")
    ax.set_aspect(aspect)
    fig.savefig(os.path.join(download_dir, thumbnail_name), dpi=100)
    fig.savefig(os.path.join(download_dir, image_name), dpi=1000)

    time.sleep(1)

    return image_name, thumbnail_name


def process_single(
    xnat_object,
    wait=1,
    verbose=True,
    raise_=False,
    pacs_address="",
    pacs_port="",
    pacs_ae="",
    xnat_ae="XNAT",
):

    try:

        # create download folder
        download_dir = os.path.join("tmp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(download_dir)

        # download
        nifti_files = {}
        reference_dcm = None
        for scan in xnat_object.scans():
            type_ = scan.attrs.get("type")
            if type_ in CONTRASTS:
                f = os.path.join(download_dir, "{}.nii.gz".format(type_))
                list_ = list(scan.resource("NIFTI").files("{}_r2s_bet*".format(type_)))
                if len(list_) == 0:
                    continue
                list_[0].get(f)
                nifti_files[type_] = f
            elif type_ == "Segmentation":
                f = os.path.join(download_dir, "segmentation.nii.gz")
                scan.resource("NIFTI").file("segmentation.nii.gz").get(f)
                nifti_files[type_] = f
            elif type_ == "T1":
                reference_dcm = os.path.join(download_dir, "reference.dcm")
                scan.resource("DICOM").files()[0].get(reference_dcm)
            else:
                continue
            if verbose:
                print("Downloaded data for type {}".format(type_))

        # check that everything is there
        error = "Could not find {} file, aborting."
        for contrast in CONTRASTS:
            if contrast not in nifti_files.keys():
                raise IOError(error.format(contrast))
        if "Segmentation" not in nifti_files.keys():
            raise IOError(error.format("segmentation"))
        if reference_dcm is None:
            raise IOError(error.format("reference DCM"))

        # resample to 0.8mm isotropic
        p = mp.Pool(min(len(nifti_files), mp.cpu_count()))
        outputs = p.map(
            resample, [(file_, contrast) for contrast, file_ in nifti_files.items()]
        )
        p.close()
        p.join()
        if verbose:
            for output in outputs:
                print(output)

        # get min and max values
        p = mp.Pool(min(len(CONTRASTS), mp.cpu_count()))
        min_max = p.map(get_min_max, [nifti_files[contrast] for contrast in CONTRASTS])
        p.close()
        p.join()
        min_max = {contrast: min_max[c] for c, contrast in enumerate(CONTRASTS)}

        # convert to dicom
        reference_dcm_data = dicom.read_file(reference_dcm)
        p = mp.Pool(min(len(CONTRASTS) + 1, mp.cpu_count()))
        outputs = p.map(
            partial(
                nifti2dicom,
                download_dir=download_dir,
                reference_dcm_data=reference_dcm_data,
            ),
            [
                (nifti_files[contrast], contrast, c)
                for c, contrast in enumerate(["Segmentation"] + list(CONTRASTS))
            ],
        )
        p.close()
        p.join()
        if verbose:
            for output in outputs:
                print(output)

        # reload segmentation and make rgb
        seg_files = sorted(os.listdir(os.path.join(download_dir, "Segmentation")))
        seg_data = []
        for f in seg_files:
            current_file = dicom.read_file(
                os.path.join(download_dir, "Segmentation", f)
            )
            current_data = current_file.pixel_array
            new_data = np.zeros(list(current_data.shape) + [3], dtype=np.float64)
            for c, class_ in enumerate(CLASSES):
                new_data[current_data == (c + 1)] = np.array(COLORS[c])
            seg_data.append(new_data)
        if verbose:
            print("Converted segmentation to RGB")

        # create overlay dicom files
        p = mp.Pool(min(len(CONTRASTS), mp.cpu_count()))
        p.map(
            partial(make_overlay_dicoms, download_dir=download_dir, seg_data=seg_data),
            [(contrast, min_max[contrast]) for contrast in CONTRASTS],
        )
        p.close()
        p.join()
        if verbose:
            print("Created segmentation overlays.")

        # upload to XNAT
        t0 = time.time()
        for c, contrast in enumerate(CONTRASTS):
            scan = xnat_object.scan("999" + str(c + 1))
            if scan.exists():
                scan.delete()
                time.sleep(wait)
            scan.create()
            scan.attrs.set("type", "{} Segmentation".format(contrast))
            scan.resource("DICOM").put_dir(
                os.path.join(download_dir, contrast + "_overlay"), overwrite=True
            )
            if verbose:
                print("Uploaded DICOMs for {} Segmentation".format(contrast))
        print("Upload: {:.3f}".format(time.time() - t0))

        # send to PACS
        try:
            for c, contrast in enumerate(CONTRASTS):

                cmd = [
                    "storescu",
                    "-aec",
                    pacs_ae,
                    "-aet",
                    xnat_ae,
                    "-v",
                    "+sd",
                    pacs_address,
                    pacs_port,
                    os.path.join(download_dir, contrast + "_overlay"),
                ]
                output = subp.check_output(cmd)
                if verbose:
                    print(output)

        except Exception as e:
            if raise_:
                raise e
            else:
                if verbose:
                    print("Could not send to PACS")
                    print(cmd)
                    print("-" * 20)
                    print(e.__repr__())
                    print("-" * 20)

        # make preview images
        p = mp.Pool(min(len(CONTRASTS), mp.cpu_count()))
        results = p.map(
            partial(make_preview_images, download_dir=download_dir), CONTRASTS
        )
        p.close()
        p.join()
        image_names, thumbnail_names = list(zip(*results))
        for c, contrast in enumerate(CONTRASTS):
            scan_number = "999" + str(c + 1)
            xnat_object.scan(scan_number).resource("SNAPSHOTS").file(
                thumbnail_names[c]
            ).insert(
                os.path.join(download_dir, thumbnail_names[c]),
                content="THUMBNAIL",
                overwrite=True,
            )
            xnat_object.scan(scan_number).resource("SNAPSHOTS").file(
                image_names[c]
            ).insert(
                os.path.join(download_dir, image_names[c]),
                content="ORIGINAL",
                overwrite=True,
            )
        if verbose:
            print("Created and uploaded preview images.")

        time.sleep(wait)

    except Exception as e:

        if raise_:
            raise e
        else:
            print(
                "Couldn't create segmentation DICOMs for experiment {}".format(
                    xnat_object
                )
            )
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
        description="Create segmentation DICOMs for XNAT experiment"
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
    parser.add_argument("--pacs_address", type=str, default="", help="PACS IP address")
    parser.add_argument("--pacs_port", type=str, default="", help="PACS port")
    parser.add_argument("--pacs_ae", type=str, default="", help="PACS AE title")
    parser.add_argument("--xnat_ae", type=str, default="XNAT", help="XNAT AE title")
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

    process(
        endpoint,
        args.wait,
        args.verbose,
        args.raiseerrors,
        pacs_address=args.pacs_address,
        pacs_port=args.pacs_port,
        pacs_ae=args.pacs_ae,
        xnat_ae=args.xnat_ae,
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
# -*- coding: utf-8 -*-
