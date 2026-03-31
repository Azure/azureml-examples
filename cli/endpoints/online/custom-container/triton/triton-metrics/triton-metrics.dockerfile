FROM nvcr.io/nvidia/tritonserver:23.09-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    python3 \
    cron \
    psmisc \
    runit &&\
    mkdir -p /var/azureml-util/prometheus_client/

RUN wget https://github.com/prometheus/prometheus/releases/download/v2.30.3/prometheus-2.30.3.linux-amd64.tar.gz && \
    tar -xzf prometheus-2.30.3.linux-amd64.tar.gz -C /var/azureml-util/prometheus_client/ && \
    mv /var/azureml-util/prometheus_client/prometheus-2.30.3.linux-amd64 /var/azureml-util/prometheus_client/client && \
    rm prometheus-2.30.3.linux-amd64.tar.gz

COPY utilities /var/azureml-util/metrics_utilities
RUN chmod +x /var/azureml-util/metrics_utilities/crontab

COPY runit /var/runit
RUN chmod +x /var/runit/*/*

RUN pip install requests torch einops accelerate transformers 

CMD ["runsvdir", "/var/runit"]