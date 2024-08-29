#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# <set_parameters>
# Set the endpoint name, appending a random number for uniqueness
ENDPOINT_NAME=vllm-openai-`echo $RANDOM`
IMAGE_TAG=azureml-examples/vllm:latest


# Set the base path for storing the endpoint configuration files
BASE_PATH=endpoints/online/custom-container/vllm/llama3-8B
ROOT_PATH=$PWD
ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)
# </set_parameters> 

# <define_helper_functions> 
# Helper function to parameterize YAML
change_vars() {
  for FILE in "$@"; do 
    TMP="${FILE}_"
    cp $FILE $TMP 
    readarray -t VARS < <(cat $TMP | grep -oP '{{.*?}}' | sed -e 's/[}{]//g'); 
    for VAR in "${VARS[@]}"; do
      sed -i "s#{{${VAR}}}#${!VAR}#g" $TMP
    done
  done
}

# </define_helper_functions> 


# <build_image>
echo $ACR_NAME
# az acr login -n ${ACR_NAME}
az acr build -t $IMAGE_TAG -f $BASE_PATH/vllm-llama3.dockerfile -r $ACR_NAME $BASE_PATH
# </build_image> 


# # <create_endpoint>
# # Create an online endpoint in Azure Machine Learning with the specified endpoint name
# az ml online-endpoint create -n $ENDPOINT_NAME
# # </create_endpoint>


# # <check_endpoint_status> 
# endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
# echo $endpoint_status
# if [[ $endpoint_status == "Succeeded" ]]
# then
#   echo "Endpoint created successfully"
# else 
#   echo "Endpoint creation failed"
#   exit 1
# fi
# # </check_endpoint_status> 

# # Create a deployment on the online endpoint, directing all traffic to this deployment
# az ml online-deployment create -n blue --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/vllm-llama3-deployment.yml --all-traffic
# # </create_deployment>

# # <check_deployment_status>
# # Check the status of the deployment to ensure it was deployed successfully
# deploy_status=`az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name vllm-llama3-deployment --query "provisioning_state" -o tsv`
# echo $deploy_status
# if [[ $deploy_status == "Succeeded" ]]
# then
#     echo "Deployment completed successfully"
# else
#     echo "Deployment failed"
#     exit 1
# fi
# # </check_deployment_status>


# # <create_deployment> 
# change_vars $BASE_PATH/vllm-llama3-deployment.yml
# az ml online-deployment create -f $BASE_PATH/vllm-llama3-deployment.yml_ --all-traffic
# # </create_deployment> 

# # <check_deployment_status> 
# deploy_status=`az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name vllm-llama3 --query "provisioning_state" -o tsv`
# echo $deploy_status
# if [[ $deploy_status == "Succeeded" ]]
# then
#     echo "Deployment completed successfully"
# else
#     echo "Deployment failed"
#     exit 1
# fi
# # </check_deployment_status> 

# # <get_endpoint_details>
# # Retrieve the access key for the online endpoint
# echo "Getting access key..."
# KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)

# # Retrieve the scoring URL for the online endpoint
# echo "Getting scoring url..."
# SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
# echo "Scoring url is $SCORING_URL"
# # </get_endpoint_details>

# # <test_endpoint> 
# curl -X POST -H "Authorization: Bearer $KEY" -T "$SERVE_PATH/Text_gen_artifacts/sample_text.txt" $SCORING_URL #???
# # </test_endpoint> 

# cleanup
