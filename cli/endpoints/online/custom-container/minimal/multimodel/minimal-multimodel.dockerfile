FROM mcr.microsoft.com/azureml/minimal-py312-inference:latest

RUN pip install pandas numpy scikit-learn joblib