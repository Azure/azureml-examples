# imports
import mlflow
import argparse
import dask_mpi

import xgboost as xgb
import dask.dataframe as dd

from distributed import Client
from adlfs import AzureBlobFileSystem

# argparse setup
parser = argparse.ArgumentParser()
parser.add_argument("--num_boost_round", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--gamma", type=float, default=0)
parser.add_argument("--max_depth", type=int, default=8)
parser.add_argument("--cpus", type=int, default=4)
args = parser.parse_args()

# distributed setup
print("initializing...")
dask_mpi.initialize(args.cpus)
client = Client()
print(client)

# get data
container_name = "malware"
storage_options = {"account_name": "azuremlexamples"}
fs = AzureBlobFileSystem(**storage_options)
files = fs.ls(f"{container_name}/processed")

# read into dataframes
print("creating dataframes...")
for f in files:
    if "train" in f:
        df_train = dd.read_parquet(f"az://{f}", storage_options=storage_options)
    elif "test" in f:
        df_test = dd.read_parquet(f"az://{f}", storage_options=storage_options)

# data processing
print("processing data...")
cols = [col for col in df_train.columns if df_train.dtypes[col] != "object"]
X = df_train[cols].drop("HasDetections", axis=1).values.persist()
y = df_train["HasDetections"].persist()

# train xgboost
print("training xgboost...")
print(client)

num_boost_round = args.num_boost_round

params = {
    "objective": "binary:logistic",
    "learning_rate": args.learning_rate,
    "gamma": args.gamma,
    "max_depth": args.max_depth,
}

dtrain = xgb.dask.DaskDMatrix(client, X, y)
model = xgb.dask.train(client, params, dtrain, num_boost_round=num_boost_round)
print(model)

# predict on test data
print("making predictions...")
print(client)
X_test = df_test[[col for col in cols if "HasDetections" not in col]].values.persist()
y_pred = xgb.dask.predict(client, model, X_test)
y_pred.to_dask_dataframe().to_csv("./outputs/predictions.csv")

# save model
print("saving model...")
mlflow.xgboost.log_model(model["booster"], "./outputs/model")
