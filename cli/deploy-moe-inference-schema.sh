#!/bin/bash
set -e

# <set_variables>
RAND=`echo $RANDOM`
ENDPOINT_NAME="endpt-moe-$RAND"
# </set_variables>

BASE_PATH=endpoints/online/managed/inference-schema

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
az ml model create -f $BASE_PATH/model.yml --set version=$RAND
# </register_model> 

echo "Creating \"standard\" deployment..."

# <create_standard_deployment>
az ml online-deployment create -f $BASE_PATH/deployment-standard.yml \
    --set model="azureml:azureml-infschema:$RAND" \
    --set endpoint_name=$ENDPOINT_NAME \
    --all-traffic
# </create_standard_deployment> 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name infsrv-standard --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
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

# Get swagger url
echo "Getting scoring url..."
SWAGGER_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query openapi_uri -o tsv )

echo "Testing scoring... "
# <test_standard_scoring>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d @$BASE_PATH/sample-inputs/standard.json $SCORING_URL
# </test_standard_scoring>

echo "Getting swagger..."
# <get_standard_swagger>
curl -H "Authorization: Bearer $KEY" $SWAGGER_URL
# </get_standard_swagger>

# <create_numpy_deployment>
az ml online-deployment create -f $BASE_PATH/deployment-numpy.yml \
    --set model="azureml:azureml-infschema:$RAND" \
    --set endpoint_name=$ENDPOINT_NAME \
    --all-traffic
# </create_numpy_deployment> 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name infsrv-numpy --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

echo "Testing scoring... "
# <test_numpy_scoring>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d @$BASE_PATH/sample-inputs/numpy.json $SCORING_URL
# </test_numpy_scoring>

echo "Getting swagger..."
# <get_numpy_swagger>
curl -H "Authorization: Bearer $KEY" $SWAGGER_URL
# </get_numpy_swagger>

# <create_pandas_deployment>
az ml online-deployment create -f $BASE_PATH/deployment-pandas.yml \
    --set model="azureml:azureml-infschema:$RAND" \
    --set endpoint_name=$ENDPOINT_NAME \
    --all-traffic
# </create_pandas_deployment> 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name infsrv-pandas --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

echo "Testing scoring... "
# <test_pandas_scoring>
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d @$BASE_PATH/sample-inputs/pandas.json $SCORING_URL
# </test_pandas_scoring>

echo "Getting swagger..."
# <get_pandas_swagger>
curl -H "Authorization: Bearer $KEY" $SWAGGER_URL
# </get_pandas_swagger>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME --no-wait
# </delete_online_endpoint>

