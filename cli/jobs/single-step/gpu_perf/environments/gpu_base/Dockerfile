FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:latest

# Build nccl test binaries
# NOTE: adding sm_35, sm_37 to support K80, M60
RUN git clone -b v2.11.0 https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=1 HPI_HOME=opt/hpcx/ompi NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70" NCCL_HOME=/usr/lib/linux*

# add nccl tests binaries (all_reduce_perf) to path
ENV PATH="/nccl-tests/build/:${PATH}"

# set variable to find nccl libs
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu:/usr/local/nccl-rdma-sharp-plugins/lib"
