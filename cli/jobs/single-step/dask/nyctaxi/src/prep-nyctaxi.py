from dask.distributed import Client
import dask.dataframe as dd
import os, uuid, time
import argparse
import mlflow
from pathlib import Path

# if running in a notebook, uncomment these 2 lines
# import sys
# sys.argv = ['']

for k, v in os.environ.items():
    if k.startswith("MLFLOW"):
        print(k, v)

parser = argparse.ArgumentParser()
parser.add_argument("--nyc_taxi_dataset")
parser.add_argument("--output_folder")
args = parser.parse_args()
dataset = args.nyc_taxi_dataset
output_path = args.output_folder

print(f"dataset location: {dataset}")
os.system(f"find {dataset}")
print(f"output location: {output_path}")

c = Client("localhost:8786")
print(c)
mlflow.log_text(str(c), "dask_cluster1")

# read in the data from the provided file dataset (which is mounted at the same
# location on all nodes of the job)
df = dd.read_csv(
    f"{dataset}/*.csv", parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
)

# as an alternative, the below would be using abfs
# but without making use of the dataset that was provided
# instead it is pulling the data from a hard-coded path on the
# default_datastore.
#
# In this case, using the v1 SDK to retrieve the credentials from the
# workspace:
#
# from azureml.core import Run
# run = Run.get_context()
# ws = run.experiment.workspace

# ds = ws.get_default_datastore()
# ACCOUNT_NAME = ds.account_name
# ACCOUNT_KEY = ds.account_key
# CONTAINER = ds.container_name

# from fsspec.registry import known_implementations
# STORAGE_OPTIONS={'account_name': ACCOUNT_NAME, 'account_key': ACCOUNT_KEY}
# df = dd.read_csv(f'abfs://{CONTAINER}/nyctaxi/*.csv',
#                 storage_options=STORAGE_OPTIONS,
#                 parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

print(df.head())
mlflow.log_text(str(df.head()), "df.head")

# list of column names that need to be re-mapped
remap = {}
remap["tpep_pickup_datetime"] = "pickup_datetime"
remap["tpep_dropoff_datetime"] = "dropoff_datetime"
remap["RatecodeID"] = "rate_code"

# create a list of columns & dtypes the df must have
must_haves = {
    "VendorID": "object",
    "pickup_datetime": "datetime64[ms]",
    "dropoff_datetime": "datetime64[ms]",
    "passenger_count": "int32",
    "trip_distance": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "rate_code": "int32",
    "payment_type": "int32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "fare_amount": "float32",
    "tip_amount": "float32",
    "total_amount": "float32",
}

query_frags = [
    "fare_amount > 0 and fare_amount < 500",
    "passenger_count > 0 and passenger_count < 6",
    "pickup_longitude > -75 and pickup_longitude < -73",
    "dropoff_longitude > -75 and dropoff_longitude < -73",
    "pickup_latitude > 40 and pickup_latitude < 42",
    "dropoff_latitude > 40 and dropoff_latitude < 42",
]
query = " and ".join(query_frags)

# helper function which takes a DataFrame partition
def clean(df_part, remap, must_haves, query):
    df_part = df_part.query(query)

    # some col-names include pre-pended spaces remove & lowercase column names
    # tmp = {col:col.strip().lower() for col in list(df_part.columns)}

    # rename using the supplied mapping
    df_part = df_part.rename(columns=remap)

    # iterate through columns in this df partition
    for col in df_part.columns:
        # drop anything not in our expected list
        if col not in must_haves:
            df_part = df_part.drop(col, axis=1)
            continue

        if df_part[col].dtype == "object" and col in [
            "pickup_datetime",
            "dropoff_datetime",
        ]:
            df_part[col] = df_part[col].astype("datetime64[ms]")
            continue

        # if column was read as a string, recast as float
        if df_part[col].dtype == "object":
            df_part[col] = df_part[col].str.fillna("-1")
            df_part[col] = df_part[col].astype("float32")
        else:
            # save some memory by using 32 bit floats
            if "int" in str(df_part[col].dtype):
                df_part[col] = df_part[col].astype("int32")
            if "float" in str(df_part[col].dtype):
                df_part[col] = df_part[col].astype("float32")
            df_part[col] = df_part[col].fillna(-1)

    return df_part


import math
from math import pi
from dask.array import cos, sin, arcsin, sqrt, floor
import numpy as np


def haversine_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    x_1 = pi / 180 * pickup_latitude
    y_1 = pi / 180 * pickup_longitude
    x_2 = pi / 180 * dropoff_latitude
    y_2 = pi / 180 * dropoff_longitude

    dlon = y_2 - y_1
    dlat = x_2 - x_1
    a = sin(dlat / 2) ** 2 + cos(x_1) * cos(x_2) * sin(dlon / 2) ** 2

    c = 2 * arcsin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r


def day_of_the_week(day, month, year):
    if month < 3:
        shift = month
    else:
        shift = 0
    Y = year - (month < 3)
    y = Y - 2000
    c = 20
    d = day
    m = month + shift + 1
    return (d + floor(m * 2.6) + y + (y // 4) + (c // 4) - 2 * c) % 7


def add_features(df):
    df["hour"] = df["pickup_datetime"].dt.hour.astype("int32")
    df["year"] = df["pickup_datetime"].dt.year.astype("int32")
    df["month"] = df["pickup_datetime"].dt.month.astype("int32")
    df["day"] = df["pickup_datetime"].dt.day.astype("int32")
    df["day_of_week"] = df["pickup_datetime"].dt.weekday.astype("int32")

    df["diff"] = df["dropoff_datetime"] - df["pickup_datetime"]

    df["pickup_latitude_r"] = (df["pickup_latitude"] // 0.01 * 0.01).astype("float32")
    df["pickup_longitude_r"] = (df["pickup_longitude"] // 0.01 * 0.01).astype("float32")
    df["dropoff_latitude_r"] = (df["dropoff_latitude"] // 0.01 * 0.01).astype("float32")
    df["dropoff_longitude_r"] = (df["dropoff_longitude"] // 0.01 * 0.01).astype(
        "float32"
    )

    # df = df.drop('pickup_datetime', axis=1)
    # df = df.drop('dropoff_datetime', axis=1)
    import numpy

    df["h_distance"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    ).astype("float32")

    df["is_weekend"] = (df["day_of_week"] > 5).astype("int32")
    return df


taxi_df = clean(df, remap, must_haves, query)
taxi_df = add_features(taxi_df)

print(c)
mlflow.log_text(str(c), "dask_cluster2")
# output_path = "./outputs/nyctaxi_processed.parquet"
print("save parquet to ", output_path)
# In this case, we are not using a mounted drive to save the job's output
# but save to the ./outputs folder, which is on the nodes local drives.
# Each node will save its partitions to the local outputs folder.
# AzureML will collect these files and copy them to the job's output location
# on blob storage.
# For this to work, the target folder needs to be created on each node.
# Using DASK's Client.run method which executes a given function on each node of the cluster:
# def create_output():
#    return Path(output_path).mkdir(parents=True, exist_ok=True)


# print("create_output", c.run(create_output))


start_time = time.time()

# make sure that the output_path exists on all nodes of the cluster.
# See above for how to create it on all cluster nodes.
taxi_df.to_parquet(output_path, engine="fastparquet")

# for debug, show output folders on all nodes
def list_output():
    return os.listdir(output_path)


print(c.run(list_output))

elapsed_time = time.time() - start_time
mlflow.log_metric("time_saving_seconds", elapsed_time)

# as an alternative: this would be using abfs, writing straight to blob storage
# output_uuid = uuid.uuid1().hex
# run.log('output_uuid', output_uuid)
# print('save parquet to ', f"abfs://{CONTAINER}/output/{output_uuid}.parquet")
# taxi_df.to_parquet(f"abfs://{CONTAINER}/output/{output_uuid}.parquet",
#                    storage_options=STORAGE_OPTIONS, engine='pyarrow')

print("done")
print(c)

os.system("ls -alg " + output_path)

# print('shutting down cluster')
# c.shutdown()
