FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# Build command
#  docker build -t ghcr.io/converged-computing/metric-fmwork:gpu .
#  docker push ghcr.io/converged-computing/metric-fmwork:gpu

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

RUN python3 -m pip install --break-system-packages vllm
RUN python3 -m pip install --break-system-packages huggingface-hub
# RUN huggingface-cli download --cache-dir ./ --local-dir-use-symlinks False --revision main --local-dir models/granite-8b ibm-granite/granite-8b-code-base-128k
WORKDIR /models/granite-8b
COPY ./models/granite-8b/config.json ./config.json
COPY ./models/granite-8b/generation_config.json ./generation_config.json
COPY ./models/granite-8b/model-00001-of-00004.safetensors ./model-00001-of-00004.safetensors
COPY ./models/granite-8b/model-00001-of-00004.safetensors ./model-00001-of-00004.safetensors
COPY ./models/granite-8b/model-00003-of-00004.safetensors ./model-00003-of-00004.safetensors
COPY ./models/granite-8b/model-00003-of-00004.safetensors ./model-00003-of-00004.safetensors
COPY ./models/granite-8b/model.safetensors.index.json ./model.safetensors.index.json
COPY ./models/granite-8b/README.md ./README.md
COPY ./models/granite-8b/special_tokens_map.json ./special_tokens_map.json
COPY ./models/granite-8b/tokenizer_config.json ./tokenizer_config.json
COPY ./models/granite-8b/tokenizer.json ./tokenizer.json
RUN git clone https://github.com/IBM/fmwork.git

# Example experiment
# ./fmwork/infer/vllm/driver --model_path models/granite-8b --input_size 1024 --output_size 1024 --batch_size 1,2,4 --tensor_parallel 1

# Install oras for saving artifacts
RUN export VERSION="1.1.0" && \
    curl -LO "https://github.com/oras-project/oras/releases/download/v${VERSION}/oras_${VERSION}_linux_amd64.tar.gz" && \
    mkdir -p oras-install/ && \
    tar -zxf oras_${VERSION}_*.tar.gz -C oras-install/ && \
    mv oras-install/oras /usr/local/bin/ && \
    rm -rf oras_${VERSION}_*.tar.gz oras-install/
