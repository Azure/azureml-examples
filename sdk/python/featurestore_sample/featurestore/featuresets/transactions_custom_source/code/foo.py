# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from typing import Union

import pandas as pd
from azureml.featurestore._utils._constants import ONLINE_ON_THE_FLY


class CustomerTransactionsTransformer:
    def transform(self, df: Union[pd.DataFrame, "pyspark.sql.DataFrame"], **kwargs) -> Union[pd.DataFrame, "pyspark.sql.DataFrame"]:
        if ONLINE_ON_THE_FLY not in kwargs or kwargs[ONLINE_ON_THE_FLY] != "true":
            #spark (offline) case
            from pyspark.sql.functions import col

            ret = df.filter(col("accountID") == "1")
        else:
            #pandas (online) case
            ret = df.query("accountID == '1'")

        return ret
