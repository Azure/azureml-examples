## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

set -e

BASE_PATH=endpoints/online/triton/multi-models
MODEL1_PATH=$BASE_PATH/models/triton/densenet_onnx/1
MODEL2_PATH=$BASE_PATH/models/triton/bidaf-9/1

# <set_endpoint_name>
export ENDPOINT_NAME=triton-multi-mir-endpt-`echo $RANDOM`
# </set_endpoint_name>

# Download densenet model
mkdir -p $MODEL1_PATH
wget https://aka.ms/densenet_onnx-model -O $MODEL1_PATH/model.onnx

# Download bidaf model
mkdir -p $MODEL2_PATH
wget https://aka.ms/bidaf-9-model -O $MODEL2_PATH/model.onnx

# <deploy>
az ml endpoint create -n $ENDPOINT_NAME -f $BASE_PATH/create-endpoint-with-deployment-mir.yml
# </deploy>

# <get_status>
az ml endpoint show -n $ENDPOINT_NAME
# </get_status>

#  check if create was successful
endpoint_status=`az ml endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then  
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

deploy_status=`az ml endpoint show --name $ENDPOINT_NAME --query "deployments[?name=='blue'].provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <get_logs>
az ml endpoint get-logs -n $ENDPOINT_NAME --deployment blue
# </get_logs>

# <get_scoring_uri>
scoring_uri=$(az ml endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
scoring_uri=${scoring_uri%/*}
# </get_scoring_uri>

# <get_token>
auth_token=$(az ml endpoint get-credentials -n $ENDPOINT_NAME --query accessToken -o tsv)
# </get_token>

# <check_status_of_triton_server>
curl --request GET $scoring_uri/v2/health/ready -H "Authorization: Bearer $auth_token"
# </check_status_of_triton_server>

# <check_list_of_models_loaded>
curl --request POST $scoring_uri/v2/repository/index -H "Authorization: Bearer $auth_token"
# </check_list_of_models_loaded>

# <check_metadata_of_model1>
curl --request GET $scoring_uri/v2/models/densenet_onnx -H "Authorization: Bearer $auth_token"
# </check_metadata_of_model1>

# <check_metadata_of_model2>
curl --request GET $scoring_uri/v2/models/bidaf-9 -H "Authorization: Bearer $auth_token"
# </check_metadata_of_model2>

# <delete_endpoint>
az ml endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint>

# <delete_model>
az ml model delete -n multi-models --version 1
# </delete_model>