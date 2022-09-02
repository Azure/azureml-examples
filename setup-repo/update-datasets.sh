#!/bin/bash
# setup datastore name

# The filename of this script for help messages
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"
ROOT_DIR=$(cd "${SCRIPT_DIR}/../" && pwd)

datastore="workspaceblobstore"

# query datastore

account=$(az ml datastore show -n $datastore --query account_name -o tsv)

account=$(echo $account|tr -d '\r')

container=$(az ml datastore show -n $datastore --query container_name -o tsv)

container=$(echo $container|tr -d '\r')


# replace storage account and container names in the YAML files

sed -i 's/account-name/'"$account"'/g' "${ROOT_DIR}"/cli/assets/dataset/cloud-folder-https.yml

sed -i 's/container-name/'"$container"'/g' "${ROOT_DIR}"/cli/assets/dataset/cloud-folder-https.yml

sed -i 's/account-name/'"$account"'/g' "${ROOT_DIR}"/cli/assets/dataset/cloud-file-https.yml

sed -i 's/container-name/'"$container"'/g' "${ROOT_DIR}"/cli/assets/dataset/cloud-file-https.yml

sed -i 's/account-name/'"$account"'/g' "${ROOT_DIR}"/cli/assets/dataset/cloud-folder-wasbs.yml

sed -i 's/container-name/'"$container"'/g' "${ROOT_DIR}"/cli/assets/dataset/cloud-folder-wasbs.yml

sed -i 's/account-name/'"$account"'/g' "${ROOT_DIR}"/cli/assets/dataset/cloud-file-wasbs.yml

sed -i 's/container-name/'"$container"'/g' "${ROOT_DIR}"/cli/assets/dataset/cloud-file-wasbs.yml

