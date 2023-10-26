# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from datetime import datetime


class CustomSourceTransformer:
    def __init__(self, **kwargs):
        self.path = kwargs.get(
            "path",
            "wasbs://data@azuremlexampledata.blob.core.windows.net/feature-store-prp/datasources/transactions-source/*.parquet",
        )

    def process(
        self, start_time: datetime, end_time: datetime, **kwargs
    ) -> "pyspark.sql.DataFrame":
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        df = spark.read.parquet(self.path)
        return df
