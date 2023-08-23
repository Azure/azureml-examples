from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AccessData").getOrCreate()

with open("1. Develop a feature set and register with managed feature store.py") as f:
    exec(f.read())

with open("2. Enable materialization and backfill feature data.py") as f:
    exec(f.read())

with open("3. Experiment and train models using features.py") as f:
    exec(f.read())

with open("4. Enable recurrent materialization and run batch inference.py") as f:
    exec(f.read())
