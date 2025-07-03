#!/bin/bash 

apt-get update && apt-get install -y git wget build-essential || yum update -y && yum install -y git wget which make

export PATH=/usr/local/go/bin:$PATH
which go || (
  rm -rf go*.tar.gz
  wget https://go.dev/dl/go1.24.4.linux-amd64.tar.gz
  rm -rf /usr/local/go && tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz
)

which nfd || (
  rm -rf /opt/node-feature-discovery
  git clone -b add-cli-export https://github.com/vsoch/node-feature-discovery /opt/node-feature-discovery
 
  cd /opt/node-feature-discovery
  go mod download && make build && mv ./bin/* /usr/local/bin
)

nfd export features --path /opt/shared/features-$(hostname).json

