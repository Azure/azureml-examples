from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AccessData").getOrCreate()

# Test folder structure
import os

for path, subdirs, files in os.walk("./"):
    for name in files:
        print(os.path.join(path, name))

with open(
    "notebooks/sdk_only/1. Develop a feature set and register with managed feature store.py"
) as f:
    exec(f.read())

with open(
    "notebooks/sdk_only/2. Enable materialization and backfill feature data.py"
) as f:
    exec(f.read())

with open("notebooks/sdk_only/3. Experiment and train models using features.py") as f:
    exec(f.read())

with open(
    "notebooks/sdk_only/4. Enable recurrent materialization and run batch inference.py"
) as f:
    exec(f.read())
