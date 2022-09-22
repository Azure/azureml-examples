set -e

pip install scikit-learn pandas joblib 

# <set_variables> 
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables> 

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)

# <setup_build_directory> 
export BASE_PATH=endpoints/online/custom-container/multimodel-minimal
mkdir -p $BASE_PATH/{code,build,test-data}
cp "$(dirname $BASE_PATH)/{multimodel-minimal*} $BASE_PATH/" 
mv $BASE_PATH/multimodel-minimal.dockerfile $BASE_PATH/build/
mv $BASE_PATH/multimodel-minimal-score.py $BASE_PATH/code/
# </setup_build_directory> 

cd $BASE_PATH

# <generate_models_and_test_data> 
python multimodel-minimal-train.py
# </generate_models_and_test_data> 

cd - 

# <build_image> 
az acr build -t azureml-examples/minimal-multimodel:1 -r $ACR_NAME -f $BASE_PATH/build/multimodel-minimal.dockerfile $BASE_PATH/build 
# </build_image> 

# <create_endpoint> 
az online-endpoint create -n $ENDPOINT_NAME -f $BASE_PATH/multimodel-minimal-endpoint.yml
# </create_endpoint> 

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

# <create_deployment> 
az online-endpoint create -e $ENDPOINT_NAME -f $BASE_PATH/multimodel-minimal-deployment.yml
# </create_deployment> 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name blue --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
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
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d @"$BASE_PATH/test-data/test-data-iris.json"  $SCORING_URL
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d @"$BASE_PATH/test-data/test-data-diabetes.json"  $SCORING_URL
# </test_online_endpoints> 

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>
