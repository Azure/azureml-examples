FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py39-cpu-inference:latest

RUN pip install numpy pip scikit-learn scipy joblib