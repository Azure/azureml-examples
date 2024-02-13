# FROM nvcr.io/nvidia/tritonserver:21.08-py3
FROM nvcr.io/nvidia/tritonserver:24.01-py3-igpu

CMD tritonserver --model-repository=/models --strict-model-config=false