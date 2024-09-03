FROM pytorch/torchserve:latest-cpu

RUN pip install transformers==4.6.0

CMD ["torchserve","--start", "--disable-token-auth", "--model-store","$AZUREML_MODEL_DIR","--models","$TORCHSERVE_MODELS","--ncs"]