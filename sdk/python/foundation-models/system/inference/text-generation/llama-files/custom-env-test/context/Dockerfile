# Use the base image that includes the necessary dependencies
FROM ghcr.io/huggingface/text-generation-inference:0.9

ENV AZUREML_MODEL_DIR=/var/azureml-model
ENV AML_APP_ROOT=/var/azureml-model/code
ENV AZUREML_ENTRY_SCRIPT=score.py
ENV DEBIAN_FRONTEND=noninteractive

# COPY $AZUREML_ENTRY_SCRIPT $AML_APP_ROOT/
ENV AZUREML_ENTRY_SCRIPT="mlflow_score_script.py"
ENV AML_APP_ROOT="/var/mlflow_resources"
COPY score.py /var/mlflow_resources/mlflow_score_script.py

RUN apt-get update -y && apt-get install vim openssh-server openssh-client -y

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# List installed packages
RUN pip list

## Delete
RUN rm requirements.txt

# Inference requirements
# COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
# RUN /var/requirements/install_system_requirements.sh && \
#     cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
#     cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
#     ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
#     rm -f /etc/nginx/sites-enabled/default
# ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=3600
EXPOSE 5001 8883 8888 80

# Stop server from starting at the very beginning itself
# We are handling server start from scoring script
ENTRYPOINT []

WORKDIR $AML_APP_ROOT

CMD ["runsvdir", "/var/runit"]
