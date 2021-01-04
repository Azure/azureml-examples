# description: deploy sklearn ridge model trained on diabetes data to AKS

# imports
import json
import time
import mlflow
import mlflow.azureml
import requests

import pandas as pd

from random import randint
from pathlib import Path
from azureml.core import Workspace
from azureml.core.webservice import AksWebservice

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path(__file__).parent

# azure ml settings
experiment_name = "sklearn-diabetes-example"

# setup mlflow tracking
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)

# get latest completed run of the training
runs_df = mlflow.search_runs()
runs_df = runs_df.loc[runs_df["status"] == "FINISHED"]
runs_df = runs_df.sort_values(by="end_time", ascending=False)
print(runs_df.head())
run_id = runs_df.at[0, "run_id"]

# create deployment configuration
aks_config = AksWebservice.deploy_configuration(
    compute_target_name="aks-cpu-deploy",
    cpu_cores=2,
    memory_gb=5,
    tags={"data": "diabetes", "method": "sklearn"},
    description="Predict using webservice",
)

# create webservice
webservice, azure_model = mlflow.azureml.deploy(
    model_uri=f"runs:/{run_id}/model",
    workspace=ws,
    deployment_config=aks_config,
    service_name="sklearn-diabetes-" + str(randint(10000, 99999)),
    model_name=experiment_name,
)

# test webservice
data = pd.read_csv(prefix.joinpath("data", "diabetes", "diabetes.csv"))

sample = data.drop(["progression"], axis=1).iloc[[0]]

query_input = sample.to_json(orient="split")
query_input = eval(query_input)
query_input.pop("index", None)

# if (key) auth is enabled, retrieve the API keys. AML generates two keys.
key1, Key2 = webservice.get_keys()

# # if token auth is enabled, retrieve the token.
# access_token, refresh_after = webservice.get_token()

# If (key) auth is enabled, don't forget to add key to the HTTP header.
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + key1,
}

# # If token auth is enabled, don't forget to add token to the HTTP header.
# headers = {'Content-Type':'application/json', 'Authorization': 'Bearer ' + access_token}

response = requests.post(
    url=webservice.scoring_uri, data=json.dumps(query_input), headers=headers
)
print(response.text)

# delete webservice
time.sleep(5)
webservice.delete()
