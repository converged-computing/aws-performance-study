
FROM ubuntu:24.04
COPY entrypoint.sh /entrypoint.sh 
COPY ./install.sh /opt/actions/install.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]


FROM golang:1.24

# docker build -t ghcr.io/converged-computing/google-performance-study:nfd-export .

RUN apt-get update && apt-get install -y git && \
    git clone -b add-cli-export https://github.com/vsoch/node-feature-discovery /opt/node-feature-discovery
    
WORKDIR /opt/node-feature-discovery

RUN go mod download && make build && mv ./bin/* /usr/local/bin


