# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from datetime import datetime
from typing import Union

import pickle
import pandas as pd
from azureml.featurestore._utils._constants import ONLINE_ON_THE_FLY


class CustomerTransactionsTransformer:
    def __init__(self, **kwargs):
        pass

    def process(self, start_time: datetime, end_time: datetime, **kwargs) -> Union[pd.DataFrame, "pyspark.sql.DataFrame"]:
        # Create dataframe
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:12:00",
                    "2020-01-01 02:24:00",
                    "2020-01-02 03:36:00",
                    "2020-01-02 04:48:00",
                    "2020-01-02 06:00:00",
                    "2020-01-03 07:12:00",
                    "2020-01-03 08:24:00",
                    "2020-01-03 09:36:00",
                    "2020-01-04 10:48:00",
                ],
                "accountID": [
                    "1",
                    "2",
                    "3",
                    "1",
                    "2",
                    "3",
                    "1",
                    "2",
                    "3",
                    "1",
                ],
                "spending": [
                    100.0,
                    200.0,
                    300.0,
                    400.0,
                    500.0,
                    600.0,
                    700.0,
                    800.0,
                    900.0,
                    1000.0,
                ],
            }
        )
        df['timestamp']= pd.to_datetime(df['timestamp'], utc=True)

        if start_time:
            df = df.query("{0} >= '{1}'".format("timestamp", pd.to_datetime(start_time)))

        if end_time:
            df = df.query("{0} < '{1}'".format("timestamp", pd.to_datetime(end_time)))

        if "on_the_fly_entities" in kwargs:
            entity_values = pickle.loads(kwargs["on_the_fly_entities"])
            output_df = pd.DataFrame()
            for entity_name, entity_value in entity_values.items():
                df = df.query("{} in @entity_value".format(entity_name))
                output_df = output_df.append(df)
            df = output_df


        if ONLINE_ON_THE_FLY not in kwargs or kwargs[ONLINE_ON_THE_FLY] != "true":
            # return spark 
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            df = spark.createDataFrame(df)

        return df
