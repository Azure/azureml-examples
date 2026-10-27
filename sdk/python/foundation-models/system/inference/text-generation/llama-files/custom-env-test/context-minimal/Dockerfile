# Use the base image that includes the necessary dependencies
FROM ghcr.io/huggingface/text-generation-inference:0.9

ENV AZUREML_MODEL_DIR=/var/azureml-model \ 
    MLFLOW_MODEL_FOLDER=mlflow_model_folder/data/model \
    AML_APP_ROOT=/var/azureml-model/code \
    AZUREML_ENTRY_SCRIPT=score.py \
    DEBIAN_FRONTEND=noninteractive \
    NUM_SHARD=1 \
    PORT=5001

RUN apt-get update -y && apt-get install vim openssh-server openssh-client -y

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# List installed packages
RUN pip list

## Delete
RUN rm requirements.txt

ENV WORKER_TIMEOUT=3600
EXPOSE 5001 8883 8888 80

# Stop server from starting at the very beginning itself
# We are handling server start from scoring script
ENTRYPOINT []

ENV MODEL_ID=${AZUREML_MODEL_DIR}/${MLFLOW_MODEL_FOLDER}/

CMD ["text-generation-launcher"]
