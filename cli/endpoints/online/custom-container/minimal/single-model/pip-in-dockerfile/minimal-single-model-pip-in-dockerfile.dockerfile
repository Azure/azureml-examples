FROM mcr.microsoft.com/azureml/minimal-py312-inference:latest

RUN pip install numpy pip scikit-learn scipy joblib