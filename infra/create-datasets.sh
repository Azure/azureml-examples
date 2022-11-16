#!/bin/bash
# this dataset is needed for the sample under cli/jobs/pipelines-with-components/basics/4d_dataset_input

# The filename of this script for help messages
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"
ROOT_DIR=$(cd "${SCRIPT_DIR}/../" && pwd)

pushd "$ROOT_DIR" > /dev/null

az ml data create -f "./cli/jobs/pipelines-with-components/basics/4d_data_input/data.yml" > /dev/null 2>&1

# This step times out. May need to use az copy to upload the dataset - TODO
# mkdir data
# wget "https://azuremlexamples.blob.core.windows.net/datasets/cifar-10-python.tar.gz"
# tar -xvzf cifar-10-python.tar.gz -C data
# az ml data create --name cifar-10-example --version 1 --set path=data

az ml data create --name cifar-10-example --version 1 --path "wasbs://cifar-10-batches@azuremlexampledata.blob.core.windows.net/" > /dev/null 2>&1



az ml data create -f "./cli/jobs/pipelines-with-components/rai_pipeline_adult_analyse/data/data_adult_test.yaml" > /dev/null 2>&1
az ml data create -f "./cli/jobs/pipelines-with-components/rai_pipeline_adult_analyse/data/data_adult_train.yaml" > /dev/null 2>&1

# </create_rai_data>

popd > /dev/null