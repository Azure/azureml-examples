#!/bin/bash
#set -e

# <set_variables>
RAND=`echo $RANDOM`
ENDPOINT_NAME="endpt-moe-$RAND"
# </set_variables>

# <create_endpoint> 
az ml online-endpoint create -n $ENDPOINT_NAME
# </create_endpoint> 

# <check_endpoint> 
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi
# </check_endpoint> 

# <get_key_and_openapi_url>
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv )

echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv )
echo "Scoring url is $SCORING_URL"

echo "Getting OpenAPI (Swagger) url..."
SWAGGER_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query openapi_uri -o tsv )
echo "OpenAPI (Swagger) url is $OPENAPI_URL"
# </get_key_and_openapi_url>

# <create_decorated_deployment>
az ml online-deployment create -f endpoints/online/managed/openapi/deployment.yml \
  --set endpoint_name=$ENDPOINT_NAME \
  --set code_configuration.code=code-decorated \
  --all-traffic
# </create_decorated_deployment> 

# <check_deployment> 
deploy_status=`az ml online-deployment show --name openapi --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi
# </check_deployment> 

echo "Testing scoring... "
# <test_decorated_scoring>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d @endpoints/online/model-1/sample-request.json $SCORING_URL
# </test_decorated_scoring>

echo "Getting swagger..."
# <get_decorated_swagger>
curl -H "Authorization: Bearer $KEY" $SWAGGER_URL
# </get_decorated_swagger>

# <create_custom_deployment>
az ml online-deployment update -f endpoints/online/managed/openapi/deployment.yml \
  --set endpoint_name=$ENDPOINT_NAME \
  --set code_configuration.code=code-custom
# </create_custom_deployment> 

# <check_deployment> 
deploy_status=`az ml online-deployment show --name openapi --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi
# </check_deployment> 

echo "Testing scoring... "
# <test_custom_scoring>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d @endpoints/online/model-1/sample-request.json $SCORING_URL
# </test_custom_scoring>

echo "Getting swagger..."
# <get_custom_swagger>
curl -H "Authorization: Bearer $KEY" $SWAGGER_URL
# </get_custom_swagger>

# <delete_custom_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME --no-wait
# </delete_custom_endpoint>