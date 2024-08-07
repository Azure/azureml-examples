FROM pytorch/torchserve:latest-cpu

CMD ["torchserve","--start", "--disable-token-auth", "--model-store","$AZUREML_MODEL_DIR/torchserve","--models","$TORCHSERVE_MODELS"]