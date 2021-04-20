FROM fastdotai/fastai:latest
RUN apt-get update -qq && apt-get install -y python3 
RUN pip install azureml-mlflow
