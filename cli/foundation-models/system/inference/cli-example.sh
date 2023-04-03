#
# Define required parameters
# Update these parameters to test deployments in your own workspace
#
subscription_id="" # Replace with your subscription ID
resource_group="" # Replace with your resource group name
workspace_name="" # Replace with your workspace name
registry_name="" # Replace with your registry name
endpoint_name="" # Replace with your endpoint name
deployment_name="" # Replace with your deployment name
model_name="" # Name of the model to be deployed
sku_name="Standard_DS2_v2" # Name of the sku(instance type) Check the model-list(can be found in the parent folder(inference)) to get the most optimal sku for your model (Default: Standard_DS2_v2)


# Set default values for subscription ID, resource group, and workspace name
az account set --subscription ${subscription_id}
az configure --defaults workspace=${workspace_name} group=${resource_group}


# Validate the existence of the model in the registry and get the latest version
model_list=$(az ml model list --name ${model_name} --registry-name ${registry_name} 2>&1)
if [[ ${model_list} == *"[]"* ]]; then
    echo "Model doesn't exist in registry. Check the model list and try again."; exit 1;
fi
version_temp=${model_list#*\"version\": \"}
version=${version_temp%%\"*}
model=$(az ml model show --name ${model_name} --version ${version} --registry-name ${registry_name} 2>&1) || {
    echo "Model ${model_name} with Version ${version} doesn't exist in registry. Check the model list and try again."; exit 1;
}

# Get the model ID
model_id_temp=${model#*\"id\": \"}
model_id=${model_id_temp%%\"*}


# Check if the endpoint already exists in the workspace
if ! az ml online-endpoint show --name ${endpoint_name} &> /dev/null ; then

# If it doesn't exist, create the endpoint
# Create the endpoint.yml file
    cat <<EOF > endpoint.yml
\$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: ${endpoint_name}
auth_mode: key
EOF

    # Trigger the endpoint creation
    echo "---Creating endpoint---"
    new_endpoint=$(az ml online-endpoint create --name ${endpoint_name} --file endpoint.yml 2>&1) || {
        echo "---Endpoint creation failed---"; 
        echo ${new_endpoint}; exit 1;
    }
    echo "--Endpoint created successfully"

else
    echo "---Endpoint already exists---"

fi


# Create the deployment.yml file
cat <<EOF > deployment.yml
\$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: default
endpoint_name: ${endpoint_name}
model: ${model_id}
instance_type: ${sku_name}
instance_count: 1
EOF

# Trigger the deployment creation
echo "---Creating deployment---"
new_deployment=$(az ml online-deployment create --name ${deployment_name} --file deployment.yml 2>&1) || {
    echo "---Deployment creation failed---";
    echo ${new_deployment}; exit 1;
}
echo "---Deployment created successfully---"


# Testing the deployment's inference if sample-request file exists
if [ -s sample-request.json ]; then
    echo "---Inference testing---"
    echo "Input: "
    cat sample-request.json
    echo -e "\nOutput: "
    az ml online-endpoint invoke --name ${endpoint_name} --deployment-name ${deployment_name} --request-file sample-request.json || {
        echo "---Inference testing failed---"; exit 1;
    }
else
    echo "---No sample request file found---"
fi


# Check if delete flags were passed
delete_all_resources=false
delete_all_files=false
ARGS=$(getopt -a --options r:f: --longoptions "delete_resources:,delete_files:" -- "$@")
eval set -- "$ARGS"
while true; do
  case "$1" in
    -r|--delete_resources)
      is_true="$2"
      if [ "$is_true" == "true" ]; then
        delete_all_resources=true
      fi
      shift 2;;
    -f|--delete_files)
      is_true="$2"
      if [ "$is_true" == "true" ]; then
        delete_all_files=true
      fi
      shift 2;;
    --)
      break;;
  esac
done

# Delete the resources created if -delete_resources flag was passed
if [ "$delete_all_resources" == true ]; then
    if [[ -v new_endpoint ]]; then
        echo "---Deleting endpoint/deployment---"
        az ml online-endpoint delete --name ${endpoint_name} --yes &> /dev/null || {
            echo "---Endpoint/Deployment deletion failed---"; exit 1;
        }
        echo "---Endpoint/Deployment deleted successfully---"
    else
        echo "---Deleting deployment---"
        az ml online-deployment delete --name ${deployment_name} --endpoint-name ${endpoint_name} --yes &> /dev/null || {
            echo "---Deployment deletion failed---"; exit 1;
        }
        echo "---Deployment deleted successfully---"
    fi
fi

# Delete the files created/downloaded if -delete_files flag was passed
if [ "$delete_all_files" == true ]; then
    echo "---Deleting files---"
    rm -rf ${model_name} endpoint.yml deployment.yml
    echo "---Files deleted successfully---"
fi