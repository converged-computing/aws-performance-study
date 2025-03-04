FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Build command
#  docker build -t ghcr.io/converged-computing/metric-gpu-burn:gpu .
#  docker push ghcr.io/converged-computing/metric-gpu-burn:gpu

RUN apt-get update && \
    apt-get -qq install -y --no-install-recommends \
        apt-utils \
        locales \
        ca-certificates \
        wget \
        man \
        git \
        flex \
        ssh \
        curl \
        sudo \
        vim \
        luarocks \
        python3-dev \
        python3-pip \
        libjansson-dev \
        jq && \
    rm -rf /var/lib/apt/lists/*

# This mimics the logic from https://github.com/wilicc/gpu-burn/blob/master/Dockerfile
RUN git clone https://github.com/wilicc/gpu-burn /build && \
    cd /build && make COMPUTE=70 CUDA_VERSION=12.8.0 IMAGE_DISTRO=ubuntu

# Install oras for saving artifacts
RUN export VERSION="1.1.0" && \
    curl -LO "https://github.com/oras-project/oras/releases/download/v${VERSION}/oras_${VERSION}_linux_amd64.tar.gz" && \
    mkdir -p oras-install/ && \
    tar -zxf oras_${VERSION}_*.tar.gz -C oras-install/ && \
    mv oras-install/oras /usr/local/bin/ && \
    rm -rf oras_${VERSION}_*.tar.gz oras-install/

WORKDIR /build

CMD ["./gpu_burn", "60"]

