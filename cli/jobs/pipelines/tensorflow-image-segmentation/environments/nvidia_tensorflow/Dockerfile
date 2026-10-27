# check release notes https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/index.html
FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

# Install dependencies missing in this container
# NOTE: container already has matplotlib==3.5.1 tqdm==4.62.0
RUN python -m pip install   azureml-defaults==1.44.0 \
                            mlflow==1.25.1 \
                            azureml-mlflow==1.44.0 \
                            pynvml==11.4.1
