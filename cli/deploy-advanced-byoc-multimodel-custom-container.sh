#!/bin/bash

set -e

# <set_parameters> 
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<ACR_NAME>"
export ACR_GROUP="<ACR_GROUP>"
# </set_parameters> 

# TODO: delete
ENDPOINT_NAME=abyoc 
ACR_NAME=valwallaceskr
ACR_GROUP=v-alwallace-test

# <download_models>
export BASE_PATH="endpoints/online/custom-container/advanced-byoc-multimodel"
rm -rf $BASE_PATH/models && mkdir -p $BASE_PATH/models/{gpt,opt}
wget -P $BASE_PATH/models/gpt https://huggingface.co/EleutherAI/gpt-neo-125M/resolve/main/pytorch_model.bin 
wget -P $BASE_PATH/models/gpt https://huggingface.co/EleutherAI/gpt-neo-125M/raw/main/{config.json,tokenizer_config.json,vocab.json,merges.txt} 
wget -P $BASE_PATH/models/opt https://huggingface.co/facebook/opt-125m/resolve/main/pytorch_model.bin 
wget -P $BASE_PATH/models/opt https://huggingface.co/facebook/opt-125m/raw/main/{config.json,tokenizer_config.json,vocab.json,merges.txt}
# </download_models> 

# <create_model> 
az ml model create --path $BASE_PATH/models -n abyoc -v 1
# </create_model> 

# <build_fastapi_image>
az acr build -f $BASE_PATH/build-fastapi/Dockerfile -r $ACR_NAME -t abyoc-fastapi:1 $BASE_PATH/build-fastapi --resource-group $ACR_GROUP
# </build_fastapi_image>

# <create_endpoint>
az ml online-endpoint create -f $BASE_PATH/abyoc-endpoint.yml -n $ENDPOINT_NAME
# </create_endpoint> 

# check if create was successful
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

# Fill deployment parameters 
cp $BASE_PATH/abyoc-deployment-fastapi.yml $BASE_PATH/abyoc-deployment-fastapi_.yml
sed -i "s/ACR_NAME/$ACR_NAME/g;" $BASE_PATH/abyoc-deployment-fastapi.yml

# <create_fastapi_deployment> 
az ml online-deployment create -f $BASE_PATH/abyoc-deployment-fastapi.yml -e $ENDPOINT_NAME --all-traffic
# </create_fastapi_deployment> 

# Check deploy status
deploy_status=`az ml online-deployment show --name abyoc-fastapi --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# Reset deployment file 
mv $BASE_PATH/abyoc-deployment-fastapi_.yml $BASE_PATH/abyoc-deployment-fastapi.yml

# Get key
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# <test_fastapi_endpoint> 
curl -d @$BASE_PATH/sample-inputs/opt-input.json -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" "${SCORING_URL}opt"
curl -d @$BASE_PATH/sample-inputs/gpt-input.json -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" "${SCORING_URL}gpt"
# </test_fastapi_endpoint>

# <build_minimal_image>
az acr build -f $BASE_PATH/build-minimal/Dockerfile -r $ACR_NAME -t abyoc-minimal:1 $BASE_PATH/build-minimal --resource-group $ACR_GROUP
# </build_minimal_image>

cp $BASE_PATH/abyoc-deployment-minimal.yml $BASE_PATH/abyoc-deployment-minimal_.yml
sed -i "s/ACR_NAME/$ACR_NAME/g;" $BASE_PATH/abyoc-deployment-minimal.yml

# <create_minimal_deployment> 
az ml online-deployment create -f $BASE_PATH/abyoc-deployment-minimal.yml -e $ENDPOINT_NAME --all-traffic
# </create_minimal_deployment> 

# Check deploy status
deploy_status=`az ml online-deployment show --name abyoc-fastapi --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# Reset deployment file 
mv $BASE_PATH/abyoc-deployment-minimal_.yml $BASE_PATH/abyoc-deployment-minimal.yml

# <delete_endpoint>
az ml online-endpoint delete -n $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>

