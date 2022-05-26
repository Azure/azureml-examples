FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20220504.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/sklearn-0.24.1
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.7 pip=20.2.4

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN pip install 'matplotlib>=3.3,<3.4' \
                'psutil>=5.8,<5.9' \
                'tqdm>=4.59,<4.60' \
                'pandas>=1.1,<1.2' \
                'scipy>=1.5,<1.6' \
                'numpy>=1.10,<1.20' \
                'ipykernel~=6.0' \
                # 'azureml-core==1.41.0.post2' \
                # 'azureml-defaults==1.41.0' \
                'azureml-mlflow==0.1.0.63658373' \
                # 'azureml-telemetry==1.41.0' \
                'scikit-learn==0.24.1' --extra-index-url https://azuremlsdktestpypi.azureedge.net/azureml-core-run-submit-tests/63658373


# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
