#!/bin/bash
#set -e

# <set_variables>
RAND=`echo $RANDOM`
ENDPOINT_NAME="endpt-moe-$RAND"
MODEL_VERSION=$RAND
ASSET_PATH=endpoints/online/model-1
# </set_variables>

BASE_PATH=endpoints/online/managed/minimal/single-model-registered

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
az ml online-endpoint create -n $ENDPOINT_NAME 
# </create_endpoint> 

# Check if endpoint was successful
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

# <register_model> 
change_vars $BASE_PATH/model.yml
az ml model create -f $BASE_PATH/model.yml_
# </register_model> 

rm $BASE_PATH/model.yml_

# <create_deployment>
change_vars $BASE_PATH/deployment.yml
az ml online-deployment create -f $BASE_PATH/deployment.yml_ --all-traffic
# </create-deployment> 

rm $BASE_PATH/deployment.yml_ 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name smr --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# Get key
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv )

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv )
echo "Scoring url is $SCORING_URL"

# <test_deployment_conda_in_dockerfile>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d @$ASSET_PATH/sample-request.json $SCORING_URL
# </test_deployment_conda_in_dockerfile>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME --no-wait
# </delete_online_endpoint>