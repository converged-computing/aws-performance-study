FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# Build command
#  docker build -t ghcr.io/converged-computing/metric-fmperf:gpu .
#  docker push ghcr.io/converged-computing/metric-fmperf:gpu
# python -m fmperf.loadgen.generate-input

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

RUN git clone https://github.com/fmperf-project/fmperf && \
    cd fmperf && \
    python3 -m pip install --break-system-packages -r ./requirements.txt

# Install kustomize
RUN curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash && \
    mv kustomize /usr/local/bin

# TGIS protobufs
RUN git clone https://github.com/IBM/text-generation-inference.git && \
    cd text-generation-inference && \
    git checkout 9b4aea86846a5131bc6f672023cae5064bf9645c && \
    cd integration_tests && \
    ln -s $(which python3) /usr/bin/python && \
    # We can't run make gen-client - newer python pip needs --break-system-packages
    pip install --break-system-packages grpcio-tools==1.62.2 mypy-protobuf==3.5.0 'types-protobuf>=3.20.4' --no-cache-dir && \
    mkdir -p text_generation_tests/pb && \
    python -m grpc_tools.protoc -I../proto --python_out=text_generation_tests/pb --grpc_python_out=text_generation_tests/pb --mypy_out=text_generation_tests/pb ../proto/generation.proto && \
    find text_generation_tests/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \; && \
    touch text_generation_tests/pb/__init__.py && \
    python3 -m pip install  . --no-cache-dir --break-system-packages

RUN cd /fmperf && \
    python3 setup.py bdist_wheel && python3 -m pip install --break-system-packages --no-cache-dir ./dist/*.whl
COPY ./environment.sh ./

# Install oras for saving artifacts
RUN export VERSION="1.1.0" && \
    curl -LO "https://github.com/oras-project/oras/releases/download/v${VERSION}/oras_${VERSION}_linux_amd64.tar.gz" && \
    mkdir -p oras-install/ && \
    tar -zxf oras_${VERSION}_*.tar.gz -C oras-install/ && \
    mv oras-install/oras /usr/local/bin/ && \
    rm -rf oras_${VERSION}_*.tar.gz oras-install/
    
WORKDIR /requests
