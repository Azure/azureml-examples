from pyspark.sql import SparkSession

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--inputs", default="")
parser.add_argument("--outputs", default="")

args, unparsed = parser.parse_known_args()

spark= SparkSession.builder.getOrCreate()
sc = spark.sparkContext

df = spark.read.csv(args.inputs)
df.write.csv(args.outputs)