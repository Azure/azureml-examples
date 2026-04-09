FROM python:3.10-slim

# Install pip dependencies
# Let azureml-core manage transitive deps (numpy, pandas, scipy)
RUN pip install 'azureml-core' \
                'azureml-defaults' \
                'azureml-mlflow' \
                'azureml-telemetry' \
                'scikit-learn==1.5.2' \
                'matplotlib' \
                'psutil' \
                'tqdm' \
                'ipykernel~=6.0'

