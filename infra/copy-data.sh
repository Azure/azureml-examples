#!/bin/bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"
ROOT_DIR=$(cd "${SCRIPT_DIR}/../" && pwd)

pushd "$ROOT_DIR" > /dev/null

# setup variables

datapath="example-data"
datastore="workspaceblobstore"

# query subscription and group

subscription=$(az account show --query id -o tsv | tail -n1 | tr -d "[:cntrl:]")

group=$(az ml workspace show --query resource_group -o tsv | tail -n1 | tr -d "[:cntrl:]")

# query principal

principal=$(az ad signed-in-user show --query objectId -o tsv | tail -n1 | tr -d "[:cntrl:]")

# query datastore

STORAGE_ACCOUNT_NAME=$(az ml datastore show -n $datastore --query account_name -o tsv | tail -n1 | tr -d "[:cntrl:]")

container=$(az ml datastore show -n $datastore --query container_name -o tsv | tail -n1 | tr -d "[:cntrl:]")

endpoint=$(az ml datastore show -n $datastore --query endpoint -o tsv | tail -n1 | tr -d "[:cntrl:]")

protocol=$(az ml datastore show -n $datastore --query protocol -o tsv | tail -n1 | tr -d "[:cntrl:]")

# build strings
destination="$protocol://${STORAGE_ACCOUNT_NAME?}.blob.$endpoint/$container/$datapath/"
echo $destination

# give access to blob container

az role assignment create \
    --role "Storage Blob Data Owner" \
    --assignee $principal \
    --scope "/subscriptions/$subscription/resourceGroups/$group/providers/Microsoft.Storage/storageAccounts/$account"

# let permissions propogate

sleep 360

azcopy login

for i in {0..1}

do
  # copy iris data
  azcopy copy "https://azuremlexamples.blob.core.windows.net/datasets/iris.csv"  $destination

  # copy diabetes data
  azcopy  copy "https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv" $destination

  # copy titanic data
  azcopy  copy "https://azuremlexamples.blob.core.windows.net/datasets/titanic.csv" $destination

  # copy mnist data
  azcopy copy "https://azuremlexamples.blob.core.windows.net/datasets/mnist" $destination --recursive=true

  # copy cifar data
  azcopy  copy "https://azuremlexamples.blob.core.windows.net/datasets/cifar-10-python.tar.gz" $destination

  # copy mltable data
  azcopy copy "https://azuremlexamples.blob.core.windows.net/datasets/mltable-sample" $destination --recursive=true

done

popd > /dev/null