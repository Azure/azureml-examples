#!/bin/bash
set -x

# setup variables

datapath="example-data"

datastore="workspaceblobstore"



# query subscription and group

subscription=$(az account show --query id -o tsv)

group=$(az ml workspace show --query resource_group -o tsv)



# query principal

principal=$(az ad signed-in-user show --query objectId -o tsv)



# query datastore

account=$(az ml datastore show -n $datastore --query account_name -o tsv)

container=$(az ml datastore show -n $datastore --query container_name -o tsv)

endpoint=$(az ml datastore show -n $datastore --query endpoint -o tsv)

protocol=$(az ml datastore show -n $datastore --query protocol -o tsv)



# build strings

destination="$protocol://$account.blob.$endpoint/$container/$datapath/"



# give access to blob container

az role assignment create \
    --role "Storage Blob Data Owner" \
    --assignee $principal \
    --scope "/subscriptions/$subscription/resourceGroups/$group/providers/Microsoft.Storage/storageAccounts/$account"



# let permissions propogate

sleep 360



for i in {0..1}

do

  # copy iris data

  azcopy cp "https://azuremlexamples.blob.core.windows.net/datasets/iris.csv" $destination



  # copy diabetes data

  azcopy cp "https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv" $destination



  # copy titanic data

  azcopy cp "https://azuremlexamples.blob.core.windows.net/datasets/titanic.csv" $destination



  # copy mnist data

  azcopy cp "https://azuremlexamples.blob.core.windows.net/datasets/mnist" $destination --recursive



  # copy cifar data

  azcopy cp "https://azuremlexamples.blob.core.windows.net/datasets/cifar-10-python.tar.gz" $destination

  
  # copy mltable data

  azcopy cp "https://azuremlexamples.blob.core.windows.net/datasets/mltable-sample" $destination --recursive

done

