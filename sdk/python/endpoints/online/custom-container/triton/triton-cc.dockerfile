FROM nvcr.io/nvidia/tritonserver:25.04-py3

CMD tritonserver --model-repository=/models --strict-model-config=false