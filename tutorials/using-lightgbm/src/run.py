# imports
import mlflow
import argparse
import dask_mpi

import lightgbm as lgbm
import dask.dataframe as dd

from distributed import Client
from adlfs import AzureBlobFileSystem

# define functions
def main(args):
    # distributed setup
    print("initializing...")
    dask_mpi.initialize(nthreads=args.cpus_per_node)
    client = Client()
    print(client)

    # get data
    print("connecting to data...")
    print(client)
    container_name = "nyctlc"
    storage_options = {"account_name": "azureopendatastorage"}
    fs = AzureBlobFileSystem(**storage_options)

    # read into dataframe
    print("creating dataframes...")
    df = dd.read_parquet(
        f"az://{container_name}/yellow/puYear=2018/puMonth=*/*.parquet",
        storage_options=storage_options,
    ).persist()

    # data processing
    print("processing data...")
    print(client)
    cols = [
        col
        for col in df.columns
        if (df.dtypes[col] != "object") and (df.dtypes[col] != "datetime64[ns]")
    ]

    X = df[cols].drop("tipAmount", axis=1).values.persist()
    y = df["tipAmount"].values.persist()

    # train lightgbm
    print("training lightgbm...")
    print(client)

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_iterations": 1000,
        "learning_rate": 0.1,
        "num_leaves": 16,
    }

    mlflow.log_params(params)  # log to the run

    model = lgbm.dask.LGBMRegressor(**params).fit(X, y)
    print(model)

    # predict on test data
    print("making predictions...")

    # save model
    print("saving model...")
    print(client)
    mlflow.lightgbm.log_model(model, "./outputs/model")


if __name__ == "__main__":
    # argparse setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus_per_node", type=int, default=4)
    args = parser.parse_args()

    # run functions
    main(args)
