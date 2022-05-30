FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

RUN sed 's/main$/main universe/' -i /etc/apt/sources.list

ARG APT_PACKAGES="wget \
    git \
    mriconvert \
    dcmtk \
    nifti2dicom \
    python \
    python3 \
    python3-pip \
    nano \
    bc \
    locales \
    zip"

RUN apt update -y && export DEBIAN_FRONTEND=noninteractive && apt upgrade -y && \
    apt install $APT_PACKAGES -y

# FSL 6.0.5.2
COPY fsl/fslinstaller.py fslinstaller.py
RUN python fslinstaller.py -q -d /usr/local/fsl

RUN apt install python3.8-venv -y

ARG USERNAME="xnat"
ARG UID="1001"
ARG GID="1001"
ARG PASSWORD="start"

ARG PIP_PACKAGES="\
    pyxnat==1.1.0.0 \
    torch \
    torchvision \
    git+https://github.com/MIC-DKFZ/batchgenerators.git \
    matplotlib \
    numpy \
    SimpleITK \
    scikit-image \
    nibabel \
    pillow \
    pydicom"

RUN useradd -m -u $UID -U -s /bin/bash $USERNAME && \
    echo "$USERNAME:$PASSWORD" | chpasswd

COPY scripts /scripts
RUN chown -R $USERNAME:$USERNAME scripts

RUN locale-gen en_US.UTF-8

USER $USERNAME

# setup venv
ENV VIRTUAL_ENV_PATH=/home/$USERNAME/venv
RUN python3 -m venv $VIRTUAL_ENV_PATH
ENV PATH="$VIRTUAL_ENV_PATH/bin:$PATH"

# Python
RUN python3 -m pip install --upgrade setuptools pip
RUN python3 -m pip install --upgrade $PIP_PACKAGES
WORKDIR /scripts/
RUN git clone https://github.com/MIC-DKFZ/HD-BET.git
RUN python3 -m pip install -e HD-BET

# nnUnet
RUN git clone https://github.com/MIC-DKFZ/nnUnet.git
RUN python3 -m pip install -e nnUnet

# HD-BET models
RUN echo "from HD_BET.utils import maybe_download_parameters\n\
for i in range(5):\n\
    maybe_download_parameters(i)"\
> hdbet_models.py && python3 hdbet_models.py
WORKDIR /

ENV FSLDIR="/usr/local/fsl"
ENV FSLWISH="/usr/local/fsl/bin/fslwish"
ENV FSLDIR="/usr/local/fsl"
ENV FSLMACHINELIST=""
ENV FSLTCLSH="/usr/local/fsl/bin/fsltclsh"
ENV FSLREMOTECALL=""
ENV FSLLOCKDIR=""
ENV FSLGECUDAQ="cuda.q"
ENV FSLOUTPUTTYPE="NIFTI_GZ"
ENV FSLMULTIFILEQUIT="TRUE"
ENV PATH="${PATH}:${FSLDIR}/bin"
ENV LANG="en_US.UTF-8"
ENV LANGUAGE="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"

ENV XNAT_HOST="" \
    XNAT_ENDPOINT="" \
    XNAT_USER="" \
    XNAT_PASS="" \
    REQUEST_WAIT=1 \
    PACS_ADDRESS="localhost" \
    PACS_PORT=11112 \
    PACS_AE="XNAT" \
    XNAT_AE="XNAT" \
    USER_NAME="admin" \
    USER_PASSWORD="admin" \
    DEVICE="0"

ENTRYPOINT ["sh", "-c",\
            "id;\
            date;\
            python3 scripts/update_scan_types.py -url ${XNAT_HOST} -s ${XNAT_ENDPOINT} -u ${USER_NAME} -p ${USER_PASSWORD} -w ${REQUEST_WAIT} -v;\
            date;\
            python3 scripts/cleanup.py -url ${XNAT_HOST} -s ${XNAT_ENDPOINT} -u ${USER_NAME} -p ${USER_PASSWORD} -w ${REQUEST_WAIT} -v;\
            date;\
            python3 scripts/dicom_to_nifti.py -url ${XNAT_HOST} -s ${XNAT_ENDPOINT} -u ${USER_NAME} -p ${USER_PASSWORD} -w ${REQUEST_WAIT} -v;\
            date;\
            python3 scripts/preprocess.py -url ${XNAT_HOST} -s ${XNAT_ENDPOINT} -u ${USER_NAME} -p ${USER_PASSWORD} -w ${REQUEST_WAIT} -v -d ${DEVICE};\
            date;\
            python3 scripts/segment.py -url ${XNAT_HOST} -s ${XNAT_ENDPOINT} -u ${USER_NAME} -p ${USER_PASSWORD} -w ${REQUEST_WAIT} -v -d ${DEVICE};\
            date;\
            python3 scripts/segmentation_to_dicom.py -url ${XNAT_HOST} -s ${XNAT_ENDPOINT} -u ${USER_NAME} -p ${USER_PASSWORD} -w ${REQUEST_WAIT} -v --pacs_address ${PACS_ADDRESS} --pacs_port ${PACS_PORT} --pacs_ae ${PACS_AE} --xnat_ae ${XNAT_AE};\
            date;\
            python3 scripts/longitudinal.py -url ${XNAT_HOST} -s ${XNAT_ENDPOINT} -u ${USER_NAME} -p ${USER_PASSWORD} -w ${REQUEST_WAIT} -v --pacs_address ${PACS_ADDRESS} --pacs_port ${PACS_PORT} --pacs_ae ${PACS_AE} --xnat_ae ${XNAT_AE};\
            date"]

CMD [""]
