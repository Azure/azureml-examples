FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

# Install pip dependencies
RUN pip install --no-cache-dir \
                'matplotlib>=3.5,<4' \
                'psutil' \
                'tqdm' \
                'pandas>=1.3,<3' \
                'scipy>=1.7,<2' \
                'numpy>=1.21,<2' \
                'ipykernel~=6.0' \
                'azureml-core' \
                'azureml-defaults' \
                'azureml-mlflow' \
                'azureml-telemetry' \
                'scikit-learn~=1.5.0' \
                'mlflow~=2.12.0'

