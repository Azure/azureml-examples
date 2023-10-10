from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AccessData").getOrCreate()

import os

for path, subdirs, files in os.walk("./"):
    for name in files:
        print(os.path.join(path, name))

print("=======Test Notebook 1============")
with open(
    "notebooks/sdk_only/1. Develop a feature set and register with managed feature store.py"
) as f:
    exec(f.read())

print("=======Test Notebook 2============")
with open("notebooks/sdk_only/2. Experiment and train models using features.py") as f:
    exec(f.read())

print("=======Test Notebook 3============")
with open(
    "notebooks/sdk_only/3. Enable recurrent materialization and run batch inference.py"
) as f:
    exec(f.read())

# # exclude 5th notebook for now
# print("=======Test Notebook 4============")
# with open("notebooks/sdk_only/4. Enable online store and run online inference.py") as f:
#     exec(f.read())
