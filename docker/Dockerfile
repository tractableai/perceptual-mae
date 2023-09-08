FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG USERNAME
ARG USER_UID
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND noninteractive

# run as non-root user
RUN apt update \
    && apt install -y sudo \
    && groupadd --gid $USER_GID $USERNAME \
    && adduser --uid $USER_UID --gid $USER_GID --no-create-home --disabled-password $USERNAME \
    && usermod -aG sudo $USERNAME \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# basic setup
RUN ln -sf /usr/share/zoneinfo/$(curl https://ipapi.co/timezone) /etc/localtime \
    && apt install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update

# install python
RUN apt install -y python3.11 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 \
    && apt install -y python3-pip python3.11-venv python3.11-dev

# install supporting packages
RUN apt install -y \
    libturbojpeg \
    ffmpeg \
    libopencv-dev \
    # mmdet requirements
    ffmpeg \
    libsm6 \
    libxext6 \
    ninja-build \
    libglib2.0-0 \
    libxrender-dev \
    # clean-up
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME