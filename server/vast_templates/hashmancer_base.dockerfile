# Hashmancer Optimized Base Image for Vast.ai
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=all
ENV PYTHONPATH=/workspace/hashmancer

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    python3-dev \
    python3-pip \
    htop \
    nvtop \
    screen \
    tmux \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit if needed
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-11-7 && \
    rm cuda-keyring_1.0-1_all.deb

# Create workspace
WORKDIR /workspace

# Pre-clone Hashmancer (this speeds up deployment)
RUN git clone https://github.com/Infernal-Insights/Hashmancer.git hashmancer

# Pre-build Darkling
WORKDIR /workspace/hashmancer/darkling
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Install Python dependencies
WORKDIR /workspace/hashmancer
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy worker scripts
COPY worker/ ./worker/

# Set up entry point
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/hashmancer
ENTRYPOINT ["/entrypoint.sh"]