# description: deploy pytorch cnn model trained on mnist data to aks

# imports
import json
import time
import mlflow
import mlflow.azureml
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import randint
from pathlib import Path
from azureml.core import Workspace
from azureml.core.webservice import AksWebservice

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# azure ml settings
experiment_name = "pytorch-mnist-mlproject-example"

# setup mlflow tracking
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

# get latest compelted run of the training
model = None
runs = ws.experiments[experiment_name].get_runs()
run = next(runs)
while run.get_status() != "Completed" or model is None:
    run = next(runs)
    try:
        model = run.register_model(experiment_name, model_path="model")
    except:
        pass

# create deployment configuration
aks_config = AksWebservice.deploy_configuration(
    compute_target_name="aks-cpu-deploy",
    cpu_cores=2,
    memory_gb=5,
    tags={"data": "mnist", "method": "pytorch"},
    description="predict using webservice",
)

# create webservice
webservice, azure_model = mlflow.azureml.deploy(
    model_uri=f"runs:/{run.id}/model",
    workspace=ws,
    deployment_config=aks_config,
    service_name="pytorch-mnist-" + str(randint(10000, 99999)),
    model_name="pytorch-mnist-example",
)

# test webservice
img = pd.read_csv(prefix.joinpath("data", "mnist", f"{randint(0, 9)}-example.csv"))
data = {"data": elem for elem in img.to_numpy().reshape(1, 1, -1).tolist()}

import matplotlib.pyplot as plt

response = webservice.run(json.dumps(data))
response = sorted(response[0].items(), key=lambda x: x[1], reverse=True)

print("predicted label:", response[0][0])
# plt.imshow(np.array(img).reshape(28, 28), cmap="gray")

# delete webservice
time.sleep(5)
webservice.delete()
