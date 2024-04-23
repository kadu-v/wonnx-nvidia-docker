FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# FROM ubuntu:22.04

# Needed to share GPU
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV XDG_RUNTIME_DIR=/tmp/runtime-xdg_runtime_dir

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV TZ Asia/Tokyo
ENV NVIDIA_DRIVER_CAPABILITIES graphics

RUN apt-get update \
    && apt-get install -y \
    sudo \
    xxd \
    wget \
    curl \
    git \
    xz-utils \
    zip \
    unzip \
    ca-certificates \
    --no-install-recommends 

# Install vulkan tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update &&\
    apt-get install -y \
    pciutils \
    vulkan-tools \
    mesa-utils \
    libglib2.0-0

# Install python and packages
RUN apt-get update &&\
    apt-get install -y python3-pip
RUN pip3 install \
    onnx==1.16.0 \
    wonnx==0.5.1 \
    opencv-python==4.9.0.80 \
    autopep8
RUN pip3 install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
    
USER $USERNAME
# Install rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup-init.sh \
    && chmod +x /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y \
    && rm -rf /tmp/rustup-init.sh
ENV PATH "$PATH:$HOME/.cargo/bin"