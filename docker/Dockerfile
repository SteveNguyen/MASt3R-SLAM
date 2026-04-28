# Self-contained dev image for MASt3R-SLAM.
#
# Why this base: nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 ships
#   - CUDA 12.8 toolkit at /usr/local/cuda (matches PyTorch cu128 wheels exactly)
#   - cuDNN 9 dev headers
#   - gcc 11 by default (well under the gcc<=14 ceiling enforced by nvcc 12.8)
# so the three CUDA extensions in this repo (mast3r_slam_backends, lietorch,
# curope) compile without any host-side toolchain juggling. cu128 is required
# for native sm_120 (Blackwell, RTX 50-series).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# System deps:
#   build-essential, ninja-build, pkg-config, git -> compile + clone lietorch
#   python3.11 (+ dev/venv) -> the version the README's conda env uses
#   libgl/libglfw/libxrender/libsm/libxext -> runtime for the imgui+moderngl viewer
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
        pkg-config \
        software-properties-common \
        wget \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
        libglfw3 libglfw3-dev \
        libegl1 libgles2 \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single static binary).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && cp /root/.local/bin/uvx /usr/local/bin/uvx

# Entrypoint lazy-runs uv sync the first time the venv volume is empty.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
