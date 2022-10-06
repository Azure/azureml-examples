FROM pytorch/torchserve:latest-cpu

CMD ["torchserve","--start","--model-store","$MODEL_BASE_PATH/torchserve","--models","densenet161=densenet161.mar"]
