FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive

# Build command
#  docker build -t ghcr.io/converged-computing/metric-deepgem:gpu .
#  docker push ghcr.io/converged-computing/metric-deepgem:gpu

RUN apt-get update && apt-get install -y git curl && apt-get clean

# Test JIT compilation
# python tests/test_jit.py

# Test all GEMM implements (normal, contiguous-grouped and masked-grouped)
# python tests/test_core.py

RUN git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git && \
    cd DeepGEMM && \
    python setup.py develop && python setup.py install

# Install oras for saving artifacts
RUN export VERSION="1.1.0" && \
    curl -LO "https://github.com/oras-project/oras/releases/download/v${VERSION}/oras_${VERSION}_linux_amd64.tar.gz" && \
    mkdir -p oras-install/ && \
    tar -zxf oras_${VERSION}_*.tar.gz -C oras-install/ && \
    mv oras-install/oras /usr/local/bin/ && \
    rm -rf oras_${VERSION}_*.tar.gz oras-install/
