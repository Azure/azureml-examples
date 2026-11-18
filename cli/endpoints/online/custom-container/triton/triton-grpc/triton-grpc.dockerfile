FROM nvcr.io/nvidia/tritonserver:21.08-py3

CMD tritonserver --model-repository=/models --strict-model-config=false