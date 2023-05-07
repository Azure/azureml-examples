# This is a pipeline is for illustration purpose only. Do not use it for production use.
import argparse
import os

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark import SparkFiles

from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

parser = argparse.ArgumentParser()
parser.add_argument("--run_id_file", type=str, help="hdfs path of observation data")
parser.add_argument("--model_name", type=str, help="hdfs path of observation data")
parser.add_argument("--evaluation_result", type=str, help="entity df time series column name")

args, _ = parser.parse_known_args()

spark.sparkContext.addFile(args.run_id_file)

run_id_file_name =os.path.basename(args.run_id_file)

with open(SparkFiles.get(run_id_file_name)) as runidFile:
    runid = runidFile.readline()


sub_id = os.environ["AZUREML_ARM_SUBSCRIPTION"]
rg = os.environ["AZUREML_ARM_RESOURCEGROUP"]
ws = os.environ["AZUREML_ARM_WORKSPACE_NAME"]

print("sub:" + sub_id + " rg:" + rg + " ws:" + ws)

from azure.ai.ml import MLClient

#connect to the workspace
ml_client = MLClient(AzureMLOnBehalfOfCredential(), sub_id, rg, ws)


from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model_path="azureml://jobs/"+ runid + "/outputs/model_output"
print(model_path)

file_model = Model(
    path=model_path,
    type=AssetTypes.CUSTOM_MODEL,
    name=args.model_name
)
ml_client.models.create_or_update(file_model)

print("Done!")
