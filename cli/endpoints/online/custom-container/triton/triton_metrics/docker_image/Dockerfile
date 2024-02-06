ARG BASE_TRITON_IMAGE=mcr.microsoft.com/mirror/nvcr/nvidia/tritonserver

ARG BASE_TRITON_TAG=23.09-py3

FROM $BASE_TRITON_IMAGE:$BASE_TRITON_TAG

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

# Conda Environment
ENV PATH=/opt/miniconda/bin:$PATH
ARG MINICONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh"
RUN wget -qO /tmp/miniconda.sh $MINICONDA_URL && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    #https://github.com/advisories/GHSA-qwmp-2cf2-g9g6
    #https://github.com/advisories/GHSA-r9hx-vwmv-q579
    conda install -c conda-forge cryptography==41.0.4 certifi=2023.07.22 requests==2.31.0 && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm -f /tmp/miniconda.sh && \
    find /opt/miniconda -type d -name __pycache__ | xargs rm -rf

COPY utilities /var/azureml-util/metrics_utilities
RUN chmod +x /var/azureml-util/metrics_utilities/crontab

COPY runit /var/runit
RUN chmod +x /var/runit/*/*

RUN echo $BASE_TRITON_IMAGE:$BASE_TRITON_TAG > /IMAGE_INFORMATION
RUN pip install torch einops accelerate transformers sentencepiece
CMD ["runsvdir", "/var/runit"]