#!/bin/bash

set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables>

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)

export BASE_PATH=endpoints/online/custom-container/minimal/single-model
export ASSET_PATH=endpoints/online/model-1

# Helper function to change parameters in yaml files
change_vars() {
  for FILE in "$@"; do 
    TMP="${FILE}_"
    cp $FILE $TMP 
    readarray -t VARS < <(cat $TMP | grep -oP '{{.*?}}' | sed -e 's/[}{]//g'); 
    for VAR in "${VARS[@]}"; do
      sed -i "s/{{${VAR}}}/${!VAR}/g" $TMP
    done
  done
}

# <create_endpoint>
change_vars $BASE_PATH/minimal-single-model-endpoint.yml
az ml online-endpoint create -f $BASE_PATH/minimal-single-model-endpoint.yml_
# </create_endpoint>

rm $BASE_PATH/minimal-single-model-endpoint.yml_

# Get key and url 
echo "Getting access key and scoring URL..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# <build_image_pip_in_dockerfile>
az acr build -f $BASE_PATH/pip-in-dockerfile/minimal-single-model-pip-in-dockerfile.dockerfile -t azureml-examples/minimal-single-model-pip-in-dockerfile:1 -r $ACR_NAME $ASSET_PATH
# </build_with_pip_in_dockerfile>

# <create_deployment_pip_in_dockerfile>
DEPLOYMENT_YML=$BASE_PATH/pip-in-dockerfile/minimal-single-model-pip-in-dockerfile-deployment.yml 
change_vars $DEPLOYMENT_YML
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f "${DEPLOYMENT_YML}_" --all-traffic
# </create_deployment_pip_in_dockerfile> 

# <test_deployment_pip_in_dockerfile>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d @$ASSET_PATH/sample-request.json $SCORING_URL
# </test_deployment_pip_in_dockerfile>

rm $BASE_PATH/pip-in-dockerfile/*.yml_

# <build_image_conda_in_dockerfile>
az acr build -f $BASE_PATH/conda-in-dockerfile/minimal-single-model-conda-in-dockerfile.dockerfile -t azureml-examples/minimal-single-model-conda-in-dockerfile:1 -r $ACR_NAME $ASSET_PATH
# </build_with_conda_in_dockerfile>

# <create_deployment_conda_in_dockerfile>
DEPLOYMENT_YML=$BASE_PATH/conda-in-dockerfile/minimal-single-model-conda-in-dockerfile-deployment.yml 
change_vars $DEPLOYMENT_YML
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f "${DEPLOYMENT_YML}_" --all-traffic
# </create_deployment_conda_in_dockerfile> 

# <test_deployment_conda_in_dockerfile>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d @$ASSET_PATH/sample-request.json $SCORING_URL
# </test_deployment_conda_in_dockerfile>

rm $BASE_PATH/conda-in-dockerfile/*.yml_

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>
