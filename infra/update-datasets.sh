#!/bin/bash
# setup datastore name

# The filename of this script for help messages
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"
ROOT_DIR=$(cd "${SCRIPT_DIR}/../" && pwd)

pushd "$ROOT_DIR" > /dev/null

datastore="workspaceblobstore"

# query datastore

account=$(az ml datastore show -n $datastore --query account_name -o tsv)

account=$(echo $account|tr -d '\r')

container=$(az ml datastore show -n $datastore --query container_name -o tsv)

container=$(echo $container|tr -d '\r')


# replace storage account and container names in the YAML files

chmod -R 755 "./cli/assets"
sed -i 's/account-name/'"$account"'/g' "./cli/assets/data/cloud-folder-https.yml"

sed -i 's/container-name/'"$container"'/g' "./cli/assets/data/cloud-folder-https.yml"

sed -i 's/account-name/'"$account"'/g' "./cli/assets/data/cloud-file-https.yml"

sed -i 's/container-name/'"$container"'/g' "./cli/assets/data/cloud-file-https.yml"

sed -i 's/account-name/'"$account"'/g' "./cli/assets/data/cloud-folder-wasbs.yml"

sed -i 's/container-name/'"$container"'/g' "./cli/assets/data/cloud-folder-wasbs.yml"

sed -i 's/account-name/'"$account"'/g' "./cli/assets/data/cloud-file-wasbs.yml"

sed -i 's/container-name/'"$container"'/g' "./cli/assets/data/cloud-file-wasbs.yml"

popd > /dev/null
