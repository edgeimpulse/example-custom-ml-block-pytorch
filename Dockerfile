# Simple Ubuntu 24.04 base image with Python3.12 and CUDA setup already (for GPU training)
FROM public.ecr.aws/g7a8t7v6/ei-custom-ml-block-base:v1.95.5

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
