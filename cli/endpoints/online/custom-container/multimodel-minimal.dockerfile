FROM mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest

RUN pip install pandas numpy scikit-learn==0.24 joblib
