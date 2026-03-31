FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Install pip dependencies
RUN pip install --no-cache-dir \
                'pandas' \
                'numpy<2' \
                'scikit-learn==1.3.2' \
                'mlflow>=2.8,<3' \
                'cloudpickle' \
                'azureml-mlflow' \
                'azureml-defaults'

