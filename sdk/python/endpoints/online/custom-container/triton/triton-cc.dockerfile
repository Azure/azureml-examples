FROM nvcr.io/nvidia/tritonserver:25.03-py3

CMD tritonserver --model-repository=/models --strict-model-config=false