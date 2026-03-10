# Adding NCCL tests on top of AzureML curated environment

# check release notes https://github.com/Azure/AzureML-Containers/blob/master/base/gpu/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04/release-notes.md
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:latest

# set env var to find nccl rdma plugins inside this container
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nccl-rdma-sharp-plugins/lib"

##############################################################################
# NCCL TESTS
##############################################################################
ENV NCCL_TESTS_TAG=v2.11.0

# NOTE: adding gencodes to support K80, M60, V100, A100
RUN mkdir /tmp/nccltests && \
    cd /tmp/nccltests && \
    git clone -b ${NCCL_TESTS_TAG} https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
        make \
            MPI=1 \
            NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80" \
            CUDA_HOME=/usr/local/cuda && \
        cp ./build/* /usr/local/bin && \
    rm -rf /tmp/nccltests

# if you want lstopo
# RUN apt update && \
#     apt install hwloc -y && \
#     rm -rf /var/lib/apt/lists/*
