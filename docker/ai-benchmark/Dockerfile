FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# Build command
#  docker build -t ghcr.io/converged-computing/metric-ai-benchmark:gpu .
#  docker push ghcr.io/converged-computing/metric-ai-benchmark:gpu

# Utilities
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
        jq && \
    rm -rf /var/lib/apt/lists/*

# Note that tensorflow-gpu has been removed
RUN python3 -m pip install --break-system-packages tensorflow && \
    python3 -m pip install --break-system-packages git+https://github.com/cloudmercato/ai-benchmark

# Install oras for saving artifacts
RUN export VERSION="1.1.0" && \
    curl -LO "https://github.com/oras-project/oras/releases/download/v${VERSION}/oras_${VERSION}_linux_amd64.tar.gz" && \
    mkdir -p oras-install/ && \
    tar -zxf oras_${VERSION}_*.tar.gz -C oras-install/ && \
    mv oras-install/oras /usr/local/bin/ && \
    rm -rf oras_${VERSION}_*.tar.gz oras-install/
