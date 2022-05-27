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

Create longitudinal analysis for XNAT subject

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
import csv
from collections import OrderedDict
from datetime import datetime
import os
import shutil
import subprocess as subp
import time
import traceback

import pydicom as dicom
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
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

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
CLASSES = ("Enhancing Tumor", "Edema / Nonenhancing Tumor")
COLORS = {
    "Enhancing Tumor": (1., 0., 0.),
    "Edema / Nonenhancing Tumor": (0., 1., 0.)
}

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


def process_single(xnat_object, wait=1, verbose=True, raise_=False, pacs_address="", pacs_port="", pacs_ae="", xnat_ae="XNAT"):

    try:

        # create download folder
        download_dir = os.path.join("tmp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(download_dir)

        # download
        volume_files = []
        dates = []
        for experiment in xnat_object.experiments():
            try:
                exp_date = experiment.attrs.get("date").replace("-", "")
                file_ = os.path.join(download_dir, exp_date + ".csv")
                experiment.scan("9990").resource("CSV").file("volumes.csv").get(file_)
                volume_files.append(file_)
                dates.append(exp_date)
            except pyxnat.core.errors.DataError:
                pass
        if verbose: print("Downloaded CSV files for {} dates".format(len(dates)))

        # sort by date
        for i, (d, v) in enumerate(sorted(zip(dates, volume_files))):
            dates[i] = d
            volume_files[i] = v

        volumes = OrderedDict()
        median_adcs = OrderedDict()
        median_cbvs = OrderedDict()

        for c in CLASSES:
            volumes[c] = []
            median_adcs[c] = []
            median_cbvs[c] = []
        for f, file_ in enumerate(volume_files):
            reader = csv.reader(open(file_, "r"))
            labels = next(reader)
            values = list(map(float, next(reader)))
            try:
                adcs = list(map(float, next(reader)))
            except StopIteration:
                adcs = None
            try:
                cbvs = list(map(float, next(reader)))
            except StopIteration:
                cbvs = None

            # if there is only one line after volumes, we don't know which it is.
            # Use threshold = 100 to differentiate
            if adcs is not None and cbvs is None:
                if np.mean(adcs) < 100:
                    cbvs = adcs
                    adcs = None

            skip_date = True
            for l, label in enumerate(labels):
                if label in CLASSES:
                    volumes[label].append(np.around(values[l], 1))
                    if adcs is not None:
                        median_adcs[label].append(adcs[l])
                    else:
                        median_adcs[label].append(None)
                    if cbvs is not None:
                        median_cbvs[label].append(cbvs[l])
                    else:
                        median_cbvs[label].append(None)
                    skip_date = False
            if skip_date:
                del dates[f]

        # download reference DICOM (first from T1 in latest experiment)
        for experiment in xnat_object.experiments():
            if experiment.attrs.get("date").replace("-", "") == dates[-1]:
                latest_experiment = experiment
                for scan in experiment.scans():
                    if scan.attrs.get("type") == "T1":
                        scan.resource("DICOM").files()[0].get(os.path.join(download_dir, "reference.dcm"))
                        break
        if verbose: print("Downloaded reference DICOM")

        def safe_div(x, y):
            if y == 0:
                return np.inf
            else:
                return x / y
        # calculate changes
        changes_to_previous = OrderedDict()
        changes_to_baseline = OrderedDict()
        for k in volumes.keys():
            changes_to_previous[k] = [np.inf]
            changes_to_baseline[k] = [np.inf]
            for i in range(1, len(volumes[k])):
                changes_to_previous[k].append(safe_div(volumes[k][i] - volumes[k][i-1], volumes[k][i-1]))
                changes_to_baseline[k].append(safe_div(volumes[k][i] - volumes[k][0], volumes[k][0]))
        # for k in volumes.keys():
        #     for i in range(len(volumes[k])):
        #         if volumes[k][i] < 0.01:
        #             volumes[k][i] = np.inf
        if verbose: print("Calculated relative changes")

        # plot
        line_styles = ["-", "--", ":", "-."]
        marker_styles = [".", ",", "o", "s", "v"]
        styles = []
        for ms in marker_styles:
            for ls in line_styles:
                styles.append((ls, ms))
        for i in range(len(dates)):
            dates[i] = " ".join([dates[i][6:], MONTHS[int(dates[i][4:6]) - 1], dates[i][:4]])

        fig = plt.figure(figsize=(20, 16), facecolor="black")

        ax = fig.add_subplot(111)
        for i, (label, values) in enumerate(volumes.items()):
            ax.plot(range(len(values)), values, color=COLORS[label], linestyle="-", marker=styles[i][1], linewidth=2)

        ax.set_ylabel("Volume [$cm^3$]", fontsize=15)
        for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(15)
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=15)
        ax.set_xlim([-0.5, len(dates) - 0.5])
        ax.set_ylim(bottom=0)
        for s in ax.spines:
            ax.spines[s].set_color("white")
            ax.spines[s].set_linewidth(2)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(axis="both", colors="white", width=2)

        disclaimer = "\
            Median ADC values calculated from raw pixel intensities, normalized CBV values from Gaussian normalized pixel intensities [median CBV tumor / SD CBV healthy white matter].\n\
            \n\
            NOT CLINICALLY VALIDATED\n\
            Do not reproduce, distribute or use (for research) without prior written permission.\n\
            Implementation of the results in a clinical setting is in the sole responsibility of the treating physician.\n\
            Questions? Contact P. Kickingereder or M. Bendszus."
        text_x = ax.get_xlim()[0] + 0.36 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        text_y = -0.7 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(text_x, text_y, disclaimer, horizontalalignment="center", verticalalignment="bottom", color="white", fontsize=15)

        lines = []
        for label in volumes.keys():
            lines.append(mpl.lines.Line2D([], [], color=COLORS[label], linestyle="-", linewidth=2))
        legend = ax.legend(labels=volumes.keys(), handles=lines, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", ncol=len(volumes.keys()), borderaxespad=0., fontsize=15)
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("black")
        for text in legend.get_texts(): text.set_color("white")

        cell_text = []
        for l, label in enumerate(volumes.keys()):
            cell_text.append([])
            for v, value in enumerate(volumes[label]):
                str_ = "{:.1f} $cm^3$\n{:+.0f}%\n{:+.0f}%".format(value, changes_to_previous[label][v] * 100, changes_to_baseline[label][v] * 100)
                str_ = str_.replace("+inf%", "--")
                str_ += "\n"
                try:
                    str_ += "{:.0f}".format(median_adcs[label][v])
                except:
                    str_ += "--"
                str_ += "\n"
                try:
                    str_ += "{:.3f}".format(median_cbvs[label][v])
                except:
                    str_ += "--"
                cell_text[l].append(str_)
        row_labels = list(volumes.keys())
        for i in range(len(row_labels)):
            row_labels[i] = row_labels[i] + "\n   change to previous exam\n   change to first exam"
            row_labels[i] = row_labels[i] + "\n   Median ADC"
            row_labels[i] = row_labels[i] + "\n   Normalized CBV"
        # bbox values depend on the label text, so adjust manually until it looks good...
        table = ax.table(cellText=cell_text, cellLoc="center", rowLabels=row_labels, bbox=[0.0, -0.53, 1, 0.42])
        for cell in table._cells:
            table._cells[cell].set_facecolor("black")
            table._cells[cell].set_fontsize(15)
            table._cells[cell].get_text().set_color("white")

        plt.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.4)
        fig.savefig(os.path.join(download_dir, "longitudinal.pdf"), bbox_extra_artists=(legend,), facecolor="black", transparent=True)
        fig.savefig(os.path.join(download_dir, "longitudinal.jpg"), bbox_extra_artists=(legend,), dpi=100, facecolor="black", transparent=True)
        fig.savefig(os.path.join(download_dir, "longitudinal_t.jpg"), bbox_extra_artists=(legend,), dpi=50, facecolor="black", transparent=True)

        if verbose: print("Saved plot")

        # create DICOM and adjust tags
        cmd = ["img2dcm", "-v", "-df", os.path.join(download_dir, "reference.dcm"),
               os.path.join(download_dir, "longitudinal.jpg"), os.path.join(download_dir, "longitudinal.dcm")]
        output = subp.check_output(cmd)

        dcm_file = dicom.read_file(os.path.join(download_dir, "longitudinal.dcm"))
        dcm_file.Modality = "SC"
        dcm_file.SeriesDescription = "Longitudinal Tumor Volumes"
        dcm_file.ProtocolName = "Longitudinal Tumor Volumes"
        dcm_file.SeriesNumber = "9990"
        dcm_file.SOPInstanceUID, dcm_file.SeriesInstanceUID = make_random_uids(dcm_file.SOPInstanceUID, dcm_file.SeriesInstanceUID)
        dcm_file.AcquisitionNumber = "1"
        dcm_file.InstanceNumber = "1"
        dcm_file.save_as(os.path.join(download_dir, "longitudinal.dcm"))

        if verbose: print(output)

        # upload
        latest_experiment.scan("9990").resource("DICOM").file("longitudinal.dcm").put(os.path.join(download_dir, "longitudinal.dcm"), overwrite=True)
        snapshot_resource = latest_experiment.scan("9990").resource("SNAPSHOTS")
        if snapshot_resource.exists(): snapshot_resource.delete()
        snapshot_resource.file("longitudinal_t.jpg").insert(os.path.join(download_dir, "longitudinal_t.jpg"), content="THUMBNAIL", overwrite=True)
        snapshot_resource.file("longitudinal.jpg").insert(os.path.join(download_dir, "longitudinal.jpg"), content="ORIGINAL", overwrite=True)
        if verbose: print("Uploaded longitudinal analysis to experiment {} of subject {}".format(latest_experiment, xnat_object))

        # send to PACS
        try:
            cmd = ["dcmsend", "-aec", pacs_ae, "-aet", xnat_ae, "-v", pacs_address, pacs_port, os.path.join(download_dir, "longitudinal.dcm")]
            output = subp.check_output(cmd)
            if verbose: print(output)
        except Exception as e:
            if raise_:
                raise e
            else:
                if verbose:
                    print("Could not send to PACS")
                    print(cmd)
                    print("-" * 20)
                    print(traceback.format_exc())
                    print(e.__repr__())
                    print("-" * 20)

        time.sleep(wait)

    except Exception as e:

        if raise_:
            raise e
        else:
            print("Couldn't create analysis for subject {}".format(xnat_object))
            print(e.__repr__())

    finally:

        # clean up
        shutil.rmtree(download_dir)
        if verbose: print("Removed all temporary data")


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

            process_single(xnat_object, wait, verbose, raise_, **kwargs)

        elif type(xnat_object) == res.Experiment:

            process(xnat_object.parent(), wait, verbose, raise_, **kwargs)

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
        description="Create longitudinal analysis for XNAT subject")
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
    parser.add_argument("--pacs_address", type=str, default="",
                        help="PACS IP address")
    parser.add_argument("--pacs_port", type=str, default="",
                        help="PACS port")
    parser.add_argument("--pacs_ae", type=str, default="",
                        help="PACS AE title")
    parser.add_argument("--xnat_ae", type=str, default="XNAT",
                        help="XNAT AE title")
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

    process(endpoint, args.wait, args.verbose, args.raiseerrors,
            pacs_address=args.pacs_address, pacs_port=args.pacs_port, pacs_ae=args.pacs_ae, xnat_ae=args.xnat_ae)

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
# -*- coding: utf-8 -*-
