FROM docker.io/tensorflow/serving:latest

ENV MODEL_NAME=hpt

COPY models /models 