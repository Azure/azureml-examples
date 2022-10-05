FROM pytorch/torchserve:latest-cpu

CMD ["torchserve","--start","--model-store","$MODEL_BASE_PATH/torchserve","--models","densenet161.mar","--ts-config","$MODEL_BASE_PATH/torchserve/config.properties"]
