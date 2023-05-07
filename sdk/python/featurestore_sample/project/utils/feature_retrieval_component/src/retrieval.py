import argparse
import os
import re
from pathlib import Path
from tempfile import mkdtemp

from azureml.dataprep.api._rslex_executor import ensure_rslex_environment
from azureml.dataprep.rslex import Copier, PyIfDestinationExists, PyLocationInfo
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

from azureml.featurestore import FeatureStoreClient
from azureml.featurestore.contracts.feature_retrieval_spec import FeatureRetrievalSpec
from azureml.featurestore import get_offline_features
from azureml.featurestore._utils._constants import FEATURE_RETRIEVAL_SPEC_YAML_FILENAME
from azureml.featurestore._utils.utils import _ensure_azureml_full_path

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

parser = argparse.ArgumentParser()
parser.add_argument("--observation_data", type=str, help="hdfs path of observation data")
parser.add_argument("--timestamp_column", type=str, help="entity df time series column name")
parser.add_argument("--input_model", required=False, type=str, help="model asset using features")
parser.add_argument("--feature_retrieval_spec", required=False, type=str, help="feature retrieval spec file")
parser.add_argument("--data_with_features", type=str, help="output path")
args, _ = parser.parse_known_args()


ensure_rslex_environment()

sub_id = os.environ["AZUREML_ARM_SUBSCRIPTION"]
rg = os.environ["AZUREML_ARM_RESOURCEGROUP"]
ws = os.environ["AZUREML_ARM_WORKSPACE_NAME"]


if args.input_model:
    feature_retrieval_spec_folder = args.input_model
else:
    feature_retrieval_spec_folder = args.feature_retrieval_spec

feature_retrieval_spec = FeatureRetrievalSpec.from_config(feature_retrieval_spec_folder)
fs_arm_uri = feature_retrieval_spec.feature_stores[0].uri
print("feature store:" + fs_arm_uri)

fs_sub_id = re.search(r'/([\w-]+)/resourcegroups', fs_arm_uri).group(1)
print(fs_sub_id) 

# extract the resource group name
fs_resource_group = re.search(r'/resourcegroups/([\w-]+)/workspaces', fs_arm_uri).group(1)
print(fs_resource_group)

# extract the workspace name
fs_name = re.search(r'/workspaces/([\w-]+)$', fs_arm_uri).group(1)
print(fs_name)


featurestore = FeatureStoreClient(
    credential = AzureMLOnBehalfOfCredential(), 
    subscription_id = fs_sub_id, 
    resource_group_name = fs_resource_group, 
    name = fs_name
)

features = featurestore.resolve_feature_retrieval_spec(feature_retrieval_spec_folder)

entity_df_path = args.observation_data
if entity_df_path.endswith("parquet"):
    entity_df = spark.read.parquet(entity_df_path)
elif entity_df_path.endswith("csv"):
    entity_df = spark.read.csv(entity_df_path, header=True)
else:
    print("Attempting to read as parquet files")
    entity_df = spark.read.parquet(entity_df_path)

training_df = get_offline_features(features = features, observation_data = entity_df, timestamp_column = args.timestamp_column)

print("Printing head of featureset...")
print(training_df.head(5))

print("Outputting dataset to parquet files.")
training_df.write.mode("overwrite").parquet(os.path.join(args.data_with_features, "data"))

# Write feature_retrieval_spec.yaml to the output_folder.
if_destination_exists = PyIfDestinationExists.MERGE_WITH_OVERWRITE
dest_uri = PyLocationInfo.from_uri(_ensure_azureml_full_path(args.data_with_features, sub_id, rg, ws))
src_uri = feature_retrieval_spec_folder + FEATURE_RETRIEVAL_SPEC_YAML_FILENAME

Copier.copy_uri(dest_uri, src_uri, if_destination_exists, "")

print("Done!")
