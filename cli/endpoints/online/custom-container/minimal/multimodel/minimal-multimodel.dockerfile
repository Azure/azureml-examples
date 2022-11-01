FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest

RUN pip install pandas numpy scikit-learn joblib