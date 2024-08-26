#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# <set_parameters>
# Set the endpoint name, appending a random number for uniqueness
ENDPOINT_NAME=vllm-openai-`echo $RANDOM`

# Set the base path for storing the endpoint configuration files
BASE_PATH=endpoints/online/custom-container/vllm/llama3-8B

# <create_endpoint>
# Create an online endpoint in Azure Machine Learning with the specified endpoint name
az ml online-endpoint create -n $ENDPOINT_NAME -f endpoints/online/custom-container/vllm/llama3-8B/vllm-llama3-endpoint.yml
# </create_endpoint>

# Create a deployment on the online endpoint, directing all traffic to this deployment
az ml online-deployment create -n blue --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/vllm-llama3-deployment.yml --all-traffic
# </create_deployment>

# <check_deployment_status>
# Check the status of the deployment to ensure it was deployed successfully
deploy_status=`az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name vllm-llama3-deployment --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
    echo "Deployment completed successfully"
else
    echo "Deployment failed"
    exit 1
fi
# </check_deployment_status>

# <get_endpoint_details>
# Retrieve the access key for the online endpoint
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)

# Retrieve the scoring URL for the online endpoint
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"
# </get_endpoint_details>

# Test the deployed model endpoint
# curl -X POST -H "Authorization: Bearer $KEY" 

# cleanup  # Clean up by removing the serve directory and deleting the endpoint

# <test_endpoint> 
echo "Uploading testing image, the scoring is..."
curl -i -H 'Content-Type: application/json' -H "Authorization: {Bearer $TOKEN}" $SCORING_URL
# </test_endpoint> 

echo "Tested successfully, cleaning up"
cleanTestingFiles

# <delete_endpoint> 
echo "Deleting endpoint..."
az ml online-endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint> 
