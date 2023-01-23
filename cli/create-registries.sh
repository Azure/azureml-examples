# Create registry-storage-options
az ml registry create -f resources/registry/registry-storage-options.yml

if [[ $? -ne 0 ]]; then
    echo_error "Failed to create registry: DemoRegistry2" >&2
    exit 1
fi

# wait for registry to be ready
sleep 120

# Delete registry
az resource delete -g $RESOURCE_GROUP_NAME -n DemoRegistry2 --resource-type "Microsoft.MachineLearningServices/registries"

if [[ $? -ne 0 ]]; then
    echo_error "Failed to delete registry: DemoRegistry2" >&2
fi

# Create demo-registry
az ml registry create -f resources/registry/registry.yml

if [[ $? -ne 0 ]]; then
    echo_error "Failed to create registry: DemoRegistry " >&2
    exit 1
fi

sleep 120

az resource delete -g $RESOURCE_GROUP_NAME -n DemoRegistry --resource-type "Microsoft.MachineLearningServices/registries"

if [[ $? -ne 0 ]]; then
    echo_error "Failed to delete registry: DemoRegistry" >&2
fi