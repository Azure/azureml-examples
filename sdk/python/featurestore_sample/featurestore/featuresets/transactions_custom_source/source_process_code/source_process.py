# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from datetime import datetime
from typing import Union

import json
import pandas as pd
from azureml.featurestore._utils._constants import ONLINE_ON_THE_FLY


class CustomerTransactionsTransformer:
    def __init__(self, **kwargs):
        print("received kwargs:")
        for k, v in kwargs.items():
            print(f"Key: {k}, Value: {v}")

    def process(
        self, start_time: datetime, end_time: datetime, **kwargs
    ) -> "pyspark.sql.DataFrame":
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, lit, to_timestamp

        spark = SparkSession.builder.getOrCreate()
        # Create dataframe
        if "source_url" not in kwargs:
            raise ValueError("source_url is required")

        df = spark.read.parquet(kwargs["source_url"])
        if start_time:
            df = df.filter(col("timestamp") >= to_timestamp(lit(start_time)))

        if end_time:
            df = df.filter(col("timestamp") < to_timestamp(lit(end_time)))

        return df
