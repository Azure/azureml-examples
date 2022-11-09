#!/bin/bash

# Query online endpoints
ONLINE_ENDPOINT_LIST=$(az ml online-endpoint list --query "[*].[name]" -o tsv)
echo "ONLINE_ENDPOINT_LIST: $ENDPOINT_LIST"

# Query left over autoscale settings created for online endpoints
AUTOSCALE_SETTINGS_LIST=$(az monitor autoscale list  --query "[*].[name]" -o tsv)

# Query batch endpoints
BATCH_ENDPOINT_LIST=$(az ml batch-endpoint list --query "[*].[name]" -o tsv)
echo "BATCH_ENDPOINT_LIST: $BATCH_ENDPOINT_LIST"

# Query storage accounts
STORAGE_ACCOUNT_LIST=$(az storage account list --query "[*].[name]" -o tsv | grep -E "oepstorage" | sort -u)
echo "STORAGE_ACCOUNT_LIST: $STORAGE_ACCOUNT_LIST"

# Query compute instances
CI_LIST=$(az ml compute list --type ComputeInstance --query "[*].[name]" -o tsv)
echo "CI_LIST: $CI_LIST"

# Query UAI
IDENTITY_LIST=$(az identity list --query "[].{name:name}" -o tsv | grep -E "oep-user-identity|my-cluster-identity" | sort -u)
echo "IDENTITY_LIST: $IDENTITY_LIST"

# Query left over autoscale settings created for online endpoints
AUTOSCALE_SETTINGS_LIST=$(az monitor autoscale list  --query "[*].[name]" -o tsv | grep -E "autoscale-" | sort -u)
echo "AUTOSCALE_SETTINGS_LIST: $AUTOSCALE_SETTINGS_LIST"

#Query Workspaces
WORKSPACE_LIST=$(az ml workspace list --query "[*].[name]" -o tsv)
echo "WORKSPACE_LIST: $WORKSPACE_LIST"

# Delete online endpoints
for i in $ONLINE_ENDPOINT_LIST; do 
    echo "Deleting online-endpoint:$i" && az ml online-endpoint delete --name $i --yes --no-wait && echo "online-endpoint delete initiated for $i" ;
done

# Batch online endpoints
for i in $BATCH_ENDPOINT_LIST; do
    echo "Deleting batch-endpoint:$i" && az ml batch-endpoint delete --name $i --yes --no-wait && echo "batch-endpoint delete initiated for $i" ;
done

# Delete storage accounts
for storage in $STORAGE_ACCOUNT_LIST; do
    if [[ $storage == *"oepstorage"* ]]; then
        echo "Deleting storage account:$storage" && az storage account delete --name $storage --yes && echo "storage account delete initiated for $storage" ;
    fi
done

# Delete compute instances
for i in $CI_LIST; do 
    echo "Deleting ComputeInstance:$i" && az ml compute delete --name $i --yes --no-wait && echo "ComputeInstance delete initiated for $i" ;
done

# Delete Identities
for name in $IDENTITY_LIST; do 
    echo "Deleting identity:$name" && az identity delete --name $name && echo "Identity $name deleted" ;
done

# Delete Autoscale monitor
for i in $AUTOSCALE_SETTINGS_LIST; do 
    echo "Deleting batch-endpoint:$i" && az monitor autoscale delete --name $i && echo "monitor autoscale $i deleted" ;
done

# Delete amlcompute
amlcompute_to_delete=(
  minimal-example
  basic-example
  mycluster
  location-example
  low-pri-example
  ssh-example
  batch-cluster
  gpu-cluster-nc6
)
for compute_name in "${amlcompute_to_delete[@]}"; do
  echo "Deleting amlcompute '$compute_name'"
  # delete compute if it does exist
  COMPUTE_EXISTS=$(az ml compute list --type amlcompute -o tsv --query "[?name=='$compute_name'][name]" |  wc -l)
  if [[ COMPUTE_EXISTS -eq 1 ]]; then
      az ml compute delete --name $compute_name --yes --no-wait
      echo "amlcompute delete initiated for $compute_name"
  else
      echo "amlcompute $compute_name does not exists"
  fi
done

# delete registry of yesterday
RegistryToBeDeleted=DemoRegistry$(date -d '-1 days' +'%m%d')
echo "Deleting registry $RegistryToBeDeleted"
az resource delete -n $RegistryToBeDeleted -g $RESOURCE_GROUP_NAME --resource-type Microsoft.MachineLearningServices/registries

#delete workpsaces created by samples
for workspace in $WORKSPACE_LIST; do
    if [[ $workspace == "mlw-basic-prod-"* ]] || [[ $workspace == "mlw-basicex-prod-"* ]]; then
        az ml workspace delete -n $workspace --yes --no-wait --all-resources
        echo "workspace delete initiated for $workspace" ;
    fi
done