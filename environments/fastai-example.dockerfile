FROM fastdotai/fastai:latest
RUN pip uninstall -y enum34 
RUN pip install aenum 
RUN pip install azureml-mlflow
