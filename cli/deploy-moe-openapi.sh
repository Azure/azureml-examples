#!/bin/bash
#set -e

# <set_variables>
RAND=`echo $RANDOM`
ENDPOINT_NAME="endpt-moe-$RAND"
BASE_PATH=endpoints/online/managed/openapi
MODEL1_PATH=endpoints/online/model-1
# </set_variables>

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

# <get_key_and_openapi_url>
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv )

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv )
echo "Scoring url is $SCORING_URL"

echo "Getting OpenAPI (Swagger) url..."
SWAGGER_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query openapi_uri -o tsv )
echo "OpenAPI (Swagger) url is $OPENAPI_URL"
# </get_key_and_openapi_url>

# <create_decorated_deployment>
az ml online-deployment create -f $BASE_PATH/decorated/deployment.yml \
  --set endpoint_name=$ENDPOINT_NAME \
  --all-traffic
# </create_decorated_deployment> 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name decorated --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

echo "Testing scoring... "
# <test_decorated_scoring>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d @$MODEL1_PATH/sample-request.json $SCORING_URL
# </test_decorated_scoring>

echo "Getting swagger..."
# <get_decorated_swagger>
curl -H "Authorization: Bearer $KEY" $SWAGGER_URL
# </get_decorated_swagger>

# <create_decorated_deployment>
az ml online-deployment create -f $BASE_PATH/custom/deployment.yml \
  --set endpoint_name=$ENDPOINT_NAME \
  --all-traffic
# </create_decorated_deployment> 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name custom --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

echo "Testing scoring... "
# <test_decorated_scoring>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d @$MODEL1_PATH/sample-request.json $SCORING_URL
# </test_decorated_scoring>

echo "Getting swagger..."
# <get_decorated_swagger>
curl -H "Authorization: Bearer $KEY" $SWAGGER_URL
# </get_decorated_swagger>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME --no-wait
# </delete_online_endpoint>