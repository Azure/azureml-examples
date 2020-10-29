FROM rapidsai/rapidsai:0.16-cuda10.2-runtime-ubuntu18.04-py3.7
RUN apt-get update && \
apt-get install -y fuse && \
source activate rapids && \
pip install azureml-mlflow && \
pip install azureml-dataprep
