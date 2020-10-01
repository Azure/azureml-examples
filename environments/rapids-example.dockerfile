FROM rapidsai/rapidsai:0.15-cuda10.2-runtime-ubuntu18.04-py3.7
RUN apt-get update && \
apt-get install -y fuse && \
apt-get install -y openmpi-bin openmpi-common openssh-client openssh-server libopenmpi2 libopenmpi-dev && \
source activate rapids && \
pip install azureml-mlflow && \
pip install azureml-dataprep
