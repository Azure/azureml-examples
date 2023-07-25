import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, to_timestamp

class MyDataSourceLoader:
    def __init__(self, kwargs):
        pass

    def process(self, start_time, end_time, kwargs):
        spark = SparkSession.builder.getOrCreate() 
        path = kwargs["k1"]

        df = spark.read.parquet(path)
        if start_time:
            df = df.filter(col('timestamp') >= to_timestamp(lit(start_time)))
        if end_time:
            df = df.filter(col('timestamp') < to_timestamp(lit(end_time)))
        return df