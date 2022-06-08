# HD-GLIO-XNAT docker container

## Software versions
The docker image built from this repository is intended to be used and tested with XNAT version 1.7.6.
You can pull the ready built image from dockerhub by `docker pull neuroradhd/hd-glio-xnat:xnat1.7.6-fsl6`.
The nvidia/cuda image this container is based on is nvidia/cuda:11.1.1-devel-ubuntu20.04.
If you need another base image for driver compatibility reasons you can edit the Dockerfile accordingly and rebuild the image locally.
If you are using another XNAT version you might have to adapt the pyxnat version. To our knowledge pyxnat to XNAT version compytibility is not limited (and not documented). 
We did however encounter issues connecting from the latest pyxnat (1.4) to XNAT 1.7.6. The pyxnat version used in this image (1.1.0.0) is confirmed working with XNAT 1.7.6 on our local system.
Several processing steps include the use of the FSL library. As we provide the respective installation script with this repository
the FSL Software License Version v6 2018 applies to this repository as well. Please see https://fsl.fmrib.ox.ac.uk/fsl/fslwiki for further details.

## Prerequisits
- XNAT itself which can be obtained at https://www.xnat.org/. 
- To inculde a docker container as a processing step into XNAT you have to install the XNAT container service plugin. See https://wiki.xnat.org/container-service/ for further details.
- As the computations inside the container must be executed on the GPU, docker must be configured to use the nvidia-container-runtime as default. See https://docs.docker.com/config/containers/resource_constraints/ and https://github.com/NVIDIA/nvidia-container-runtime for details on how to enable GPU access inside docker containers. Setting `"default-runtime": "nvidia"` in /etc/docker/daemon.json is the critical requirement to enable GPU access as default for all container launches which is required for this container to work in XNAT.
- Required image series for segmentation are pre-contrast T1-, post-contrast T1-, T2- and FLAIR weighted 2D sequences in axial slice orientation, or 3D isometric volume images. The  detection of the respective contrast of each series in an examination is determined in scripts/update_scan_types.py of this repository. The definition for each contrast is kept quite general, it may however be possible that the individual sequences at your local site are not recognized properly if the respective DICOM tags differ from the given defintions too much. In this case will have to edit the specifications in update_scan_types.py accordingsly and rebuild the image.

## Setup
1. Pull or build the container on your local system. If you build locally, note that the FSL installation can be time consuming, therefore the container may take up to or even longer than one hour to build. 
2. Make sure all prerequisits are met.
3. In XNAT go to Administer -> Plugin Settings -> Images & Commands. 
4. Click Add New Command.
5. Paste the contents of command.json located in the root directory of this repository into the editor.
6. To recieve DICOM files sent from a PACS node over the network, your XNAT instance has to be present as a PACS node as well. It will therefore require an AE title set by your local system administrator. Edit the respective entry in command.json to match that title (xnat-ae -> default-value).
7. If you intend to automatically send back segmentation results to your local PACS, edit the respective entries (pacs-ae, pacs-address, pacs-port) in command.json as well.
8. Save the command script.
9. Enable the command script at Administer -> Plugin Settings -> Command Automation by clicking Add New Command Automation. Select your Project, set On Event to "On Scan Archive" and select "glioblastoma_segmentation_wrapper" in the Run Command dropdown menu.
10. Open an examination in XNAT. In the Actions menu, at the Run Containers field select "Run glioblastoma segmentation" to test the container on the current exam.

## Troubleshooting
Many steps are involved in the complete processing workflow. For problems regarding XNAT, Docker including GPU access and the container integration in XNAT, please refer to their respective documentation. 
Container execution itself can be monitored by checking the container's log files. 
See output of `docker ps` or `docker ps -a` to get the container ID. Watch the log by `docker logs <container-id>`.

## Contributions
The main contributions to this project have been made by the project owners / contributors to nnUnet, HD-BET, HD-GLIO and HD-GLIO-AUTO, in particular Fabian Isensee and Jens Petersen as parts of the Division of Medical Image Computing, German Cancer Research Center DKFZ, Heidelberg, Germany.
See https://github.com/MIC-DKFZ/nnUNet, https://github.com/NeuroAI-HD/HD-GLIO-AUTO, https://github.com/NeuroAI-HD/HD-GLIO, https://github.com/NeuroAI-HD/HD-BET.
The python scripts in this repository in particular have been authored by Jens Petersen, see authorship information in the respective python files. 
The repository has been updated and reviewed for publication by Hagen Meredig, Neuroradiology Department, University Clinic Heidelberg, Heidelberg, Germany as well as the previously mentioned contributors.

## License
This software is released under the Apache License Version 2.0, January 2004 (see LICENSE file in the root directory). 
As this software package includes the FSL library, it is also subject to the FSL Software License Version v6 2018 (see LICENSE file in the fsl directory).

