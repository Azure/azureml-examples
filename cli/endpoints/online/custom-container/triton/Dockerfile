#FROM mcr.microsoft.com/azureml/tritonserver-21.02-py38-inference
FROM mcr.microsoft.com/azureml/tritonserver-inference

# Install packages for score.py
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt 

# Set the model directory and app root
ENV AML_APP_ROOT=/var/azureml-app 
ENV AZUREML_MODEL_DIR=/var/azureml-app/azureml-models
COPY models $AZUREML_MODEL_DIR/models/triton

# Copy scoring script and label files
COPY score.py $AML_APP_ROOT
COPY densenet_labels.txt $AML_APP_ROOT
ENV AZUREML_ENTRY_SCRIPT=score.py

#ENTRYPOINT ["runsvdir", "/var/runit"]
