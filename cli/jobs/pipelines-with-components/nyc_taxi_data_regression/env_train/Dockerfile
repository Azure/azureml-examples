FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Install pip dependencies
RUN pip install --no-cache-dir \
                'matplotlib' \
                'psutil' \
                'tqdm' \
                'pandas' \
                'scipy' \
                'numpy<2' \
                'ipykernel~=6.0' \
                'azureml-core' \
                'azureml-defaults' \
                'azureml-mlflow' \
                'azureml-telemetry' \
                'scikit-learn==1.3.2'

