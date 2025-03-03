FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# Build command
#  docker build -t ghcr.io/converged-computing/metric-deepgem:gpu .
#  docker push ghcr.io/converged-computing/metric-deepgem:gpu

# Test JIT compilation
# python tests/test_jit.py

# Test all GEMM implements (normal, contiguous-grouped and masked-grouped)
# python tests/test_core.py

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

RUN git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git && \
    # Make symbolic links for third-party (CUTLASS and CuTe) include directories
    cd DeepGEMM && \
    python3 setup.py develop && python3 setup.py install

WORKDIR /DeepGEMM

# Install oras for saving artifacts
RUN export VERSION="1.1.0" && \
    curl -LO "https://github.com/oras-project/oras/releases/download/v${VERSION}/oras_${VERSION}_linux_amd64.tar.gz" && \
    mkdir -p oras-install/ && \
    tar -zxf oras_${VERSION}_*.tar.gz -C oras-install/ && \
    mv oras-install/oras /usr/local/bin/ && \
    rm -rf oras_${VERSION}_*.tar.gz oras-install/
