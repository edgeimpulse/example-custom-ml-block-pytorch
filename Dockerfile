ARG UBUNTU_VERSION=24.04

ARG ARCH=
ARG CUDA=12.9.1
ARG CUDA_SHORT=12.9
ARG CUDA_PACKAGE_VERSION=12-9
ARG CUDA_FLAVOR=base
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION} AS base
ARG CUDA
ARG CUDA_SHORT
ARG CUDA_PACKAGE_VERSION
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python, pip, and dos2unix (as when you check out install_cuda.sh on Windows it converts to CRLF which bash does not like in the next step)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip dos2unix && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA/cuDNN runtime libraries needed by TensorFlow on x86.
COPY dependencies/install_cuda.sh ./install_cuda.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    dos2unix ./install_cuda.sh && \
    /bin/bash ./install_cuda.sh && \
    rm install_cuda.sh && \
    rm -rf /var/lib/apt/lists/*

# Copy Python requirements in and install them (--break-system-packages is required if we don't use a venv).
# Installing torch from PyPI with dependencies pulls the full CUDA wheel set. The base image already
# provides most CUDA 12.9 libraries, so install the CUDA 12.9 torch wheel without dependencies and add
# only the CUDA wheel libraries that libtorch still needs at runtime.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages \
        onnx==1.22.0 \
        filelock \
        typing-extensions \
        'setuptools>=77.0.3' \
        'sympy>=1.13.3' \
        'networkx>=2.5.1' \
        jinja2 \
        'fsspec>=0.8.5' && \
    pip3 install --break-system-packages --no-deps \
        --index-url https://download.pytorch.org/whl/cu129 \
        'torch==2.13.0+cu129' && \
    pip3 install --break-system-packages \
        'cuda-toolkit[cufile,cupti]==12.9.1' \
        'nvidia-cusparselt-cu12==0.8.1' \
        'nvidia-nccl-cu12==2.29.7' \
        'nvidia-nvshmem-cu12==3.4.5'

# # Install CMake (separate script as this requires a different command on M1 Macs)
# COPY dependencies/install_cmake.sh install_cmake.sh
# RUN /bin/bash install_cmake.sh && \
#     rm install_cmake.sh

# RUN apt update && apt install -y protobuf-compiler

# # Copy Python requirements in and install them
# COPY requirements.txt ./
# RUN pip3 install -r requirements.txt

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["python3", "-u", "train.py"]
