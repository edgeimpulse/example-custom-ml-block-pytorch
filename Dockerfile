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

# Use --no-deps because requirements.txt pins the exact runtime set; letting torch resolve dependencies
# pulls the full CUDA wheel stack instead of the smaller CUDA 12.9 set used here.
COPY requirements_torch_2.13.0_cuda.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --no-deps -r requirements_torch_2.13.0_cuda.txt

# Additional user-provided packages
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages -r requirements.txt

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["python3", "-u", "train.py"]
