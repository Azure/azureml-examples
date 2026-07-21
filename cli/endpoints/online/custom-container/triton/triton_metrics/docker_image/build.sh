#!/bin/bash

set -ex

SRC="../../../src"

export IMAGE_NAME=${1:-triton-metrics}

echo "Prune unused images/containers/networks and so on ..."

docker system prune -f

docker image prune -a -f

docker build -t $IMAGE_NAME \
    -f Dockerfile .