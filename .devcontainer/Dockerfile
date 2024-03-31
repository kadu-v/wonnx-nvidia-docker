FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# FROM ubuntu:22.04

# Needed to share GPU
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV XDG_RUNTIME_DIR=/tmp/runtime-xdg_runtime_dir

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
