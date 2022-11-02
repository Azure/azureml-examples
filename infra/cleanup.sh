#!/bin/bash

for i in $(az ml online-endpoint list | jq -r .[].name); do echo "Deleting online-endpoint:$i" && az ml online-endpoint delete --name $i --yes --no-wait && echo "online-endpoint delete initiated for $i" ;done

for i in $(az ml batch-endpoint list | jq -r .[].name); do echo "Deleting batch-endpoint:$i" && az ml batch-endpoint delete --name $i --yes --no-wait && echo "batch-endpoint delete initiated for $i" ;done

for i in $(az storage account list | jq -r '.[].name' | grep "oepstorage"); do echo "Deleting storage account:$i" && az storage account delete --name $i --yes && echo "storage account delete initiated for $i" ;done

for i in $(az ml compute list --type ComputeInstance | jq -r .[].name); do echo "Deleting ComputeInstance:$i" && az ml compute delete --name $i --yes --no-wait && echo "ComputeInstance delete initiated for $i" ;done

for i in $(az identity list | jq -r '.[].name' | grep -E "oep-user-identity|my-cluster-identity"); do echo "Deleting identity:$i" && az identity delete --name $i && echo "Identity $i deleted" ;done

for i in $(az monitor autoscale list | jq -r .[].name); do echo "Deleting batch-endpoint:$i" && az monitor autoscale delete --name $i && echo "monitor autoscale $i deleted" ;done

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
az resource delete -n DemoRegistry$(date -d '-1 days' +'%m%d') --resource-type Microsoft.MachineLearningServices/registries