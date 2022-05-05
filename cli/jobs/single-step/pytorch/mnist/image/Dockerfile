FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20220418.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/pytorch-1.9-mlflow
# Create conda environment
COPY conda.yml /tmp/pytorch-1.9-mlflow.yml
RUN conda env create -f /tmp/pytorch-1.9-mlflow.yml -p $AZUREML_CONDA_ENVIRONMENT_PATH

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

RUN pip uninstall azureml-dataprep-rslex -y
RUN pip install -U azureml-dataprep-rslex --index-url=https://dataprepdownloads.azureedge.net/pypi/test-M3ME5B1GMEM3SW0W/62172131/ --no-deps
