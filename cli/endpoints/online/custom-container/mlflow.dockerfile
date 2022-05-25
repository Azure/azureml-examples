FROM mcr.microsoft.com/azureml/mlflow-ubuntu18.04-py37-cpu-inference:latest

ENV MLFLOW_MODEL_FOLDER=model 

EXPOSE 5001
CMD ["runsvdir", "/var/runit"]