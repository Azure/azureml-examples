# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from datetime import datetime


class CustomSourceTransformer:
    def __init__(self, **kwargs):
        self.path = kwargs.get("source_path")
        self.timestamp_column_name = kwargs.get("timestamp_column_name")
        if not self.path:
            raise Exception("`source_path` is not provided")
        if not self.timestamp_column_name:
            raise Exception("`timestamp_column_name` is not provided")

    def process(
        self, start_time: datetime, end_time: datetime, **kwargs
    ) -> "pyspark.sql.DataFrame":
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, lit, to_timestamp

        spark = SparkSession.builder.getOrCreate()
        df = spark.read.json(self.path)

        if start_time:
            df = df.filter(
                col(self.timestamp_column_name) >= to_timestamp(lit(start_time))
            )

        if end_time:
            df = df.filter(
                col(self.timestamp_column_name) < to_timestamp(lit(end_time))
            )

        return df
