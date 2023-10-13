from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AccessData").getOrCreate()

print("=======Test CLI Vnet Notebook============")
with open(
    "notebooks/sdk_and_cli/network_isolation/Network Isolation for Feature store.py"
) as f:
    exec(f.read())
