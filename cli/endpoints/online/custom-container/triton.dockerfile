FROM nvcr.io/nvidia/tritonserver:20.11-py3

RUN pip install Pillow
CMD tritonserver --model-repository=$MODEL_BASE_PATH/triton --strict-model-config=false --log-verbose=1