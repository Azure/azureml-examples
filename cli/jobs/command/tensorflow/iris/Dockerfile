FROM python:3.8

RUN pip install \
  mlflow \
  azureml-mlflow \
  numpy \
  scipy \
  pandas \
  scikit-learn \
  matplotlib \
  adlfs \
  fsspec \
  fastparquet \
  pyarrow \
  python-snappy \
  lz4 \
  xgboost \
  lightgbm \
  tensorflow \
  tensorflow_decision_forests

RUN apt-get update && apt-get install -y fuse liblttng-ust0

CMD ["bash"]
