FROM mcr.microsoft.com/azureml/tritonserver-21.02-py38-inference

# Set the model directory
ENV AZUREML_MODEL_DIR=/var/azureml-app/azureml-models
COPY models $AZUREML_MODEL_DIR

ENTRYPOINT [ "runsvdir", "/var/runit" ]