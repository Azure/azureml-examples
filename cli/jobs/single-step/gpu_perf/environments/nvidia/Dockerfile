# Adding NCCL tests on top of nvidia/pytorch container

# check release notes https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html
FROM nvcr.io/nvidia/pytorch:22.02-py3

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
            MPI=1 MPI_HOME=/opt/hpcx/ompi \
            NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80" \
            CUDA_HOME=/usr/local/cuda && \
        cp ./build/* /usr/local/bin && \
    rm -rf /tmp/nccltests


# if you want lstopo
# RUN apt update && \
#     apt install hwloc -y && \
#     rm -rf /var/lib/apt/lists/*
