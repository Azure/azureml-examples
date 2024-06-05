#!/bin/bash

if [ $# -eq 0 ]; then
    echo "This script deletes computes provided in the first parameter."
    echo "Comutes may include batch endpoints, compute clustrs and"
    echo "online endpoints."
    echo "Usage $0 [list of computs to delete]"
    exit 1
fi

for compute_name in "$@"
do
    COMPUTE_EXISTS=$(az ml compute list --type amlcompute -o tsv --query "[?name=='$compute_name'][name]" |  wc -l)
    if [ $COMPUTE_EXISTS -eq 1 ]; then
        echo "Deleting amlcompute '$compute_name'"
        az ml compute delete --name $compute_name --yes --no-wait
    else
        echo "Compute '$compute_name' was cleaned up by the notebook."
    fi
done
exit 0