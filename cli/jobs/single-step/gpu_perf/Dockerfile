FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:latest

# NOTE: adding sm_35, sm_37 to support K80, M60
RUN git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=1 HPI_HOME=opt/hpcx/ompi NVCC_GENCODE="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_37,code=sm_37" NCCL_HOME=/usr/lib/linux*
