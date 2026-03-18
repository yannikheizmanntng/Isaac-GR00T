FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# System dependencies + Python 3.10
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    apt-get install -y \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        netcat dnsutils \
        libgl1-mesa-glx git git-lfs libvulkan-dev \
        zip unzip wget curl \
        build-essential cmake ninja-build \
        vim less sudo htop ca-certificates man tmux \
        ffmpeg \
        libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel
RUN python3.10 -m pip install gpustat wandb==0.19.0

WORKDIR /workspace

# Copy package metadata for dependency caching
COPY pyproject.toml README.md ./

# Minimal stub so setuptools can resolve the dynamic version without the real source
RUN mkdir -p gr00t && \
    printf 'VERSION = "0.0.0"\n' > gr00t/version.py && \
    touch gr00t/__init__.py gr00t/py.typed

# Heavy torch stack — separate layer, rarely changes
RUN python3.10 -m pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    numpy==1.26.4

# All remaining deps — torch is already at the right version so pip skips it
RUN python3.10 -m pip install ".[base]"

# Replace headless OpenCV with the GL-enabled build
RUN python3.10 -m pip uninstall -y opencv-python opencv-python-headless || true && \
    python3.10 -m pip install opencv-python==4.8.0.74

RUN python3.10 -m pip install "accelerate>=0.26.0"

# Copy source — this layer is invalidated on every code change
COPY gr00t /workspace/gr00t
COPY extension /workspace/extension
COPY scripts /workspace/scripts
COPY getting_started /workspace/getting_started
COPY Makefile /workspace/Makefile
# Register the package as editable; deps already installed above
RUN python3.10 -m pip install -e . --no-deps

CMD ["python3.10", "-u", "extension/job_server.py"]
