#!/bin/bash
# this dataset is needed for the sample under cli/jobs/pipelines-with-components/basics/4d_dataset_input

# The filename of this script for help messages
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"
ROOT_DIR=$(cd "${SCRIPT_DIR}/../" && pwd)

pushd "$ROOT_DIR" > /dev/null

az ml data create -f "./cli/jobs/pipelines-with-components/basics/4d_data_input/data.yml"

# <download_untar_cifar>

mkdir data

wget "https://azuremlexamples.blob.core.windows.net/datasets/cifar-10-python.tar.gz"

tar -xvzf cifar-10-python.tar.gz -C data

# </download_untar_cifar>


# <create_cifar>


# This step times out. May need to use az copy to upload the dataset - TODO
# az ml data create --name cifar-10-example --version 1 --set path=data


# </create_cifar>


# <cleanup_cifar>

rm cifar-10-python.tar.gz

rm -r data


# </cleanup_cifar>

# <create_rai_data>

az ml data create -f "./cli/jobs/pipelines-with-components/rai_pipeline_adult_analyse/data/data_adult_test.yaml"
az ml data create -f "./cli/jobs/pipelines-with-components/rai_pipeline_adult_analyse/data/data_adult_train.yaml"

# </create_rai_data>

popd > /dev/null