FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest

ARG MODEL_NAME=model-1

USER dockeruser

# Conda is already installed
ENV CONDA_ENV_DIR=/opt/miniconda/envs

# We will pre-install the conda environment at build time
# Alternatively, the AZUREML_EXTRA_CONDA_YAML variable can be set for dynamic installation
# or can be added alongside the image in the environment yaml definition for Azure to build 

# Create a new conda environment and install the same version of the server
COPY $MODEL_NAME/environment/conda.yml /tmp/conda.yaml
RUN conda env create -n userenv -f /tmp/conda.yaml && \
    export SERVER_VERSION=$(pip show azureml-inference-server-http | grep Version | sed -e 's/.*: //')  && \ 
    $CONDA_ENV_DIR/userenv/bin/pip install azureml-inference-server-http==$SERVER_VERSION

# Update environment variables
ENV AZUREML_CONDA_ENVIRONMENT_PATH="$CONDA_ENV_DIR/userenv" 
ENV PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH" 
ENV LD_LIBRARY_PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH"
