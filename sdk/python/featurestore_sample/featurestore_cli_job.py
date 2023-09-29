from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AccessData").getOrCreate()

import os

for path, subdirs, files in os.walk("./"):
    for name in files:
        print(os.path.join(path, name))

print("=======Test CLI Notebook 1============")
with open(
    "notebooks/sdk_and_cli/1. Develop a feature set and register with managed feature store.py"
) as f:
    exec(f.read())

print("=======Test CLI Notebook 2============")
with open(
    "notebooks/sdk_and_cli/2. Enable materialization and backfill feature data.py"
) as f:
    exec(f.read())

print("=======Test CLI Notebook 3============")
with open(
    "notebooks/sdk_and_cli/3. Experiment and train models using features.py"
) as f:
    exec(f.read())

print("=======Test CLI Notebook 4============")
with open(
    "notebooks/sdk_and_cli/4. Enable recurrent materialization and run batch inference.py"
) as f:
    exec(f.read())
