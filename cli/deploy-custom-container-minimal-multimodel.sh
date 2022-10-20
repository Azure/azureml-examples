set -e

# <set_variables> 
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables> 

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)

export BASE_PATH=endpoints/online/custom-container/minimal/multimodel
export ASSET_PATH=endpoints/online/custom-container/minimal/multimodel/models

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
change_vars $BASE_PATH/minimal-multimodel-endpoint.yml
az ml online-endpoint create -n $ENDPOINT_NAME -f $BASE_PATH/minimal-multimodel-endpoint.yml_
# </create_endpoint> 

rm $BASE_PATH/*.yml_

# Check if endpoint was successful
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

# <build_image> 
az acr build -t azureml-examples/minimal-multimodel:1 -r $ACR_NAME -f $BASE_PATH/minimal-multimodel.dockerfile $BASE_PATH
# </build_image> 

# <create_deployment> 
change_vars $BASE_PATH/minimal-multimodel-deployment.yml
az ml online-deployment create -e $ENDPOINT_NAME -f $BASE_PATH/minimal-multimodel-deployment.yml_ --all-traffic
az ml online-deployment update -e $ENDPOINT_NAME -f $BASE_PATH/minimal-multimodel-deployment.yml_
# </create_deployment> 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name minimal-multimodel --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
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
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# <test_online_endpoints> 
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d @"$BASE_PATH/test-data/iris-test-data.json"  $SCORING_URL
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d @"$BASE_PATH/test-data/diabetes-test-data.json"  $SCORING_URL
# </test_online_endpoints> 

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>
