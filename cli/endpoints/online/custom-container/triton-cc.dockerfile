FROM mcr.microsoft.com/azureml/tritonserver-inference

RUN pip install -U azureml-inference-server-http[all] && \ 
    pip install pillow numpy 

ENV AML_APP_ROOT=/var/azureml-app \ 
    AZUREML_ENTRY_SCRIPT=score.py \
    AZUREML_MODEL_DIR=/var/azureml-app/azureml-models

COPY code $AML_APP_ROOT
COPY models/ $AZUREML_MODEL_DIR/models/triton

CMD ["runsvdir", "/var/runit"]