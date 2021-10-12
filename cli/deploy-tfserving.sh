## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <initialize_variables>
BASE_PATH=endpoints/online/custom-container
AML_MODEL_NAME=tfserving-mounted
MODEL_NAME=half_plus_two
MODEL_BASE_PATH=/var/azureml-app/azureml-models/$AML_MODEL_NAME/1
ENDPOINT_NAME=tfserving-endpoint
DEPLOYMENT_NAME=tfserving-deployment
# </initialize_variables>

# <download_and_unzip_model>
wget https://aka.ms/half_plus_two-model -O $BASE_PATH/half_plus_two.tar.gz
tar -xvf $BASE_PATH/half_plus_two.tar.gz -C $BASE_PATH
# </download_and_unzip_model>

# Clean up utility
cleanup(){
    rm $BASE_PATH/half_plus_two.tar.gz
    rm -r $BASE_PATH/half_plus_two
}

# <run_image_locally_for_testing>
docker run --rm -d -v $PWD/$BASE_PATH:$MODEL_BASE_PATH -p 8501:8501 \
 -e MODEL_BASE_PATH=$MODEL_BASE_PATH -e MODEL_NAME=$MODEL_NAME \
 --name="tfserving-test" docker.io/tensorflow/serving:latest
sleep 10
# </run_image_locally_for_testing>

# <check_liveness_locally>
curl -v http://localhost:8501/v1/models/$MODEL_NAME
# </check_liveness_locally>

# <check_scoring_locally>
curl --header "Content-Type: application/json" \
  --request POST \
  --data @$BASE_PATH/sample_request.json \
  http://localhost:8501/v1/models/$MODEL_NAME:predict
# </check_scoring_locally>

# <stop_image>
docker stop tfserving-test
# </stop_image>

# <create_endpoint>
az ml online-endpoint create -f $BASE_PATH/$ENDPOINT_NAME.yml
# </create_endpoint>

# <create_deployment>
az ml online-deployment create --name $DEPLOYMENT_NAME --endpoint $ENDPOINT_NAME --file $BASE_PATH/$DEPLOYMENT_NAME.yml --all-traffic
# </create_deployment>

# <get_status>
az ml online-endpoint show -n $ENDPOINT_NAME
# </get_status>

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

deploy_status=`az ml online-deployment show --name $DEPLOYMENT_NAME --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# Test remotely
echo "Testing endpoint"
for i in {1..10}
do
   # <invoke_endpoint>
   RESPONSE=$(az ml online-endpoint invoke --name $ENDPOINT_NAME --request-file $BASE_PATH/sample_request.json)
   # </invoke_endpoint>
done

echo "Tested successfully, response was $RESPONSE. Cleaning up..."

# <delete_endpoint>
az ml online-endpoint delete --name $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>

# <delete_model>
az ml model delete --name $MODEL_NAME --version 1
# </delelte_model>

# <delete_environment>
az ml environment delete --name tfserving --version 1
# </delete_environment>

cleanup