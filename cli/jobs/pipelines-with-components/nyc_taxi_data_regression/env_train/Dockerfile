FROM python:3.10.14

# Install pip dependencies
# Pinned to align with the AzureML curated MLflow no-code inference image
# (numpy 2.x ABI, scikit-learn 1.5).
RUN pip install 'matplotlib>=3.8,<3.9' \
                'psutil>=5.9,<6.0' \
                'tqdm>=4.66,<5.0' \
                'pandas>=2.2,<2.3' \
                'scipy>=1.13,<1.14' \
                'numpy>=2.0,<2.3' \
                'ipykernel~=6.0' \
                'azureml-core' \
                'azureml-defaults' \
                'azureml-mlflow' \
                'azureml-telemetry' \
                'scikit-learn==1.5.2'

