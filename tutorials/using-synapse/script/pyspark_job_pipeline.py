from pyspark.sql import SparkSession
from pyspark.sql.types import * 
import sys
import os

from azureml.core import Workspace, Datastore, Dataset
from azureml.core import Run, Dataset

spark= SparkSession.builder.getOrCreate()

run_context = Run.get_context()
dataset = run_context.input_datasets['synapseinput']

sdf = dataset.to_spark_dataframe()

sdf.show()
sdf.coalesce(1).write\
.option("header", "true")\
.csv(os.environ['synapse_step_output'],mode='overwrite')
