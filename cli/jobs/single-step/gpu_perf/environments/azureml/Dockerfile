# Adding NCCL tests on top of AzureML curated environment

FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu121-py38-torch210:latest
 
# set env var to find nccl rdma plugins inside this container /usr/mpi/gcc/openmpi-4.1.5a1
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
            NVCC_GENCODE="-gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90" \
            CUDA_HOME=/usr/local/cuda\
            MPI_HOME=/usr/local/mpi && \
        cp ./build/* /usr/local/bin && \
    rm -rf /tmp/nccltests
 
# for lstopo
RUN apt update && \
    apt install hwloc -y && \
    rm -rf /var/lib/apt/lists/*
