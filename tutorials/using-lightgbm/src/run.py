# imports
import mlflow
import argparse

import pandas as pd
#import lightgbm as lgbm
import dask.dataframe as dd

from distributed import Client
from dask_mpi import initialize
from adlfs import AzureBlobFileSystem

## TODO: remove
from dask_lightgbm import LGBMRegressor

# argparse setup
parser = argparse.ArgumentParser()
parser.add_argument("--boosting", type=str, default="gbdt")
parser.add_argument("--num_iterations", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--num_leaves", type=int, default=31)
args = parser.parse_args()

# distributed setup
print("initializing...")
initialize()
c = Client()
print(c)

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

# train lightgbm
print("training lightgbm...")
print(c)

params = {
    "objective": "binary",
    "boosting": args.boosting,
    "num_iterations": args.num_iterations,
    "learning_rate": args.learning_rate,
    "num_leaves": args.num_leaves,
}

model = LGBMRegressor(**params)
model.fit(X, y)
print(model)

# predict on test data
print("making predictions...")
print(c)
X_test = df_test[[col for col in cols if "HasDetections" not in col]].values.persist()
y_pred = model.predict(X_test)
y_pred.to_dask_dataframe().to_csv("./outputs/predictions.csv")
