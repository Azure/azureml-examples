FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20230620.v1

ENV CONDA_ENVIRONMENT_PATH /azureml-envs/llama

# Prepend path to AzureML conda environment
ENV PATH $CONDA_ENVIRONMENT_PATH/bin:$PATH

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_ENVIRONMENT_PATH -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean -a -y

RUN pip freeze

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH