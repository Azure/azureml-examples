import os
import sys
import azureml.core
from pyspark.sql import SparkSession
from azureml.core import Run, Dataset
import os, uuid, time
import argparse
import mlflow
from pathlib import Path

print(azureml.core.VERSION)
print(os.environ)
# if running in a notebook, uncomment these 2 lines
# import sys
# sys.argv = ['']

for k, v in os.environ.items():
    if k.startswith("MLFLOW"):
        print(k, v)

parser = argparse.ArgumentParser()
parser.add_argument("--nyc_taxi_dataset")
args = parser.parse_args()
dataset = args.nyc_taxi_dataset
print(f"dataset location: {dataset}")
# os.system(f"find {dataset}")

spark = (
    SparkSession.builder.appName("AML Dataprep")
    .config("spark.executor.cores", 1)
    .config("spark.executor.instances", 16)
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)

# # Azure storage access info
# blob_account_name = "azureopendatastorage"
# blob_container_name = "nyctlc"
# blob_relative_path = "yellow"
# blob_sas_token = r""

# # # Allow SPARK to read from Blob remotely
# wasbs_path = 'wasbs://%s@%s.blob.core.windows.net/%s' % (blob_container_name, blob_account_name, blob_relative_path)
# spark.conf.set(
#   'fs.azure.sas.%s.%s.blob.core.windows.net' % (blob_container_name, blob_account_name),
#   blob_sas_token)
# print('Remote blob path: ' + wasbs_path)

# SPARK read parquet, note that it won't load any data yet by now
# df = spark.read.parquet(wasbs_path)
df = spark.read.parquet(dataset)
# df.show()

print(df.head())
mlflow.log_text(str(df.head()), "df.head")

# create a list of columns & dtypes the df must have
must_haves = {
    "vendorID": "string",
    "pickupDatetime": "datetime",
    "dropoffDatetime": "datetime",
    "passengerCount": "int",
    "tripDistance": "double",
    "startLon": "double",
    "startLat": "double",
    "rateCode": "int",
    "paymentType": "int",
    "endLon": "double",
    "endLat": "double",
    "fareAmount": "double",
    "tipAmount": "double",
    "totalAmount": "double",
}

query_frags = [
    "fareAmount > 0 and fareAmount < 500",
    "passengerCount > 0 and passengerCount < 6",
    "startLon > -75 and startLon < -73",
    "endLon > -75 and endLon < -73",
    "startLat > 40 and startLat < 42",
    "endLat > 40 and endLat < 42",
]
query = " AND ".join(query_frags)


def clean(df_part, must_haves, query):
    df_part = df_part.filter(query)

    # some col-names include pre-pended spaces remove & lowercase column names
    # tmp = {col:col.strip().lower() for col in list(df_part.columns)}

    # rename using the supplied mapping
    # df_part = df_part.rename(columns=remap)

    # iterate through columns in this df partition
    for col in df_part.columns:
        # drop anything not in our expected list
        if col not in must_haves:
            df_part = df_part.drop(col)
            continue

        if df_part.select(col).dtypes[0][1] is "string" and col in [
            "pickup_datetime",
            "dropoff_datetime",
        ]:
            df_part.withColumn(col, df_part[col].cast("timestamp"))
            continue

        # if column was read as a string, recast as float
        if df_part.select(col).dtypes[0][1] is "string":
            df_part.na.fill(value=-1, subset=[col])
            df_part.withColumn(col, df_part[col].cast("double"))
        else:
            # save some memory by using 32 bit floats
            if "int" in str(df_part[col].dtype):
                df_part.withColumn(col, df_part[col].cast("int"))
            if "double" in str(df_part[col].dtype):
                df_part.withColumn(col, df_part[col].cast("double"))
            df_part.na.fill(value=-1, subset=[col])

    return df_part


import math
from math import pi, cos, sin, sqrt
import numpy as np
from pyspark.sql.functions import datediff, to_date, lit
from pyspark.sql import functions as F
from pyspark.sql.functions import *


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

    c = 2 * np.arcsin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return float(c) * r


def add_features(df):
    df = df.withColumn("hour", hour(df["pickupDatetime"]).cast("int"))
    df = df.withColumn("year", year(df["pickupDatetime"]).cast("int"))
    df = df.withColumn("month", month(df["pickupDatetime"]).cast("int"))
    df = df.withColumn("day", dayofmonth(df["pickupDatetime"]).cast("int"))
    df = df.withColumn("day_of_week", dayofweek(df["pickupDatetime"]).cast("int"))

    df = df.withColumn(
        "diff", datediff(df["dropoffDatetime"], df["pickupDatetime"]).cast("int")
    )

    df = df.withColumn(
        "startLatr", (F.floor(df["startLat"] / (0.01)) * 0.01).cast("double")
    )
    df = df.withColumn(
        "startLonr", (F.floor(df["startLon"] / (0.01)) * 0.01).cast("double")
    )
    df = df.withColumn(
        "endLatr", (F.floor(df["endLat"] / (0.01)) * 0.01).cast("double")
    )
    df = df.withColumn(
        "endLonr", (F.floor(df["endLon"] / (0.01)) * 0.01).cast("double")
    )

    # df = df.drop('pickup_datetime', axis=1)
    # df = df.drop('dropoff_datetime', axis=1)

    import numpy

    # df.withColumn("h_distance",haversine_distance(
    #     df.select("startLat"),
    #     df.select("startLon"),
    #     df.select("endLat"),
    #     df.select("endLon"),
    # ).cast('double'))

    df = df.withColumn("is_weekend", (df["day_of_week"] > 5).cast("int"))
    return df


df = (
    df.withColumnRenamed("tpepPickupDateTime", "pickupDatetime")
    .withColumnRenamed("tpepDropoffDateTime", "dropoffDatetime")
    .withColumnRenamed("rateCodeId", "rateCode")
)
taxi_df = clean(df, must_haves, query)
taxi_df = add_features(taxi_df)

start_time = time.time()

output_path = "./outputs/nyctaxi_processed.parquet"
print("save parquet to ", output_path)

print(taxi_df.head())
mlflow.log_text(str(taxi_df.head()), "taxi_df.head")

# for debug, show output folders on all nodes
def list_output():
    return os.listdir(output_path)


elapsed_time = time.time() - start_time
mlflow.log_metric("time_saving_seconds", elapsed_time)

# as an alternative: this would be using abfs, writing straight to blob storage
# output_uuid = uuid.uuid1().hex
# run.log('output_uuid', output_uuid)
# print('save parquet to ', f"abfs://{CONTAINER}/output/{output_uuid}.parquet")
# taxi_df.to_parquet(f"abfs://{CONTAINER}/output/{output_uuid}.parquet",
#                    storage_options=STORAGE_OPTIONS, engine='pyarrow')

print("done")

# make sure that the output_path exists on all nodes of the cluster.
# See above for how to create it on all cluster nodes.
taxi_df.coalesce(1).write.option("header", "true").mode("append").parquet(output_path)
