## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <initialize_variables>
BASE_PATH=endpoints/online/custom-container
AML_MODEL_NAME=tfserving-mounted
MODEL_NAME=half_plus_two
MODEL_BASE_PATH=/var/azureml-app/azureml-models/$AML_MODEL_NAME/1
ENDPOINT_NAME=tfserving-endpoint
DEPLOYMENT_NAME=tfserving
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

# Check endpoint existence
EXISTS=$(az ml endpoint show -n $ENDPOINT_NAME --query name -o tsv)

# endpoint exists, update it
if [[ $EXISTS == $ENDPOINT_NAME ]]
then 
  echo "endpoint exists, updating..."
  az ml endpoint update -f $BASE_PATH/$ENDPOINT_NAME.yml -n $ENDPOINT_NAME
else
  # <create_endpoint>
  az ml endpoint create -f $BASE_PATH/$ENDPOINT_NAME.yml -n $ENDPOINT_NAME
  # </create_endpoint>
fi

STATE=$(az ml endpoint show -n $ENDPOINT_NAME --query deployments[0].provisioning_state -o tsv)

if [[ $STATE != "Succeeded" ]]
then
  az ml endpoint get-logs -n $ENDPOINT_NAME --deployment $DEPLOYMENT_NAME
  az ml endpoint get-logs -n $ENDPOINT_NAME --deployment $DEPLOYMENT_NAME --container storage-initializer
  echo "deleting endpoint, state is "$STATE
  # <delete_endpoint_and_model>
  az ml endpoint delete -n $ENDPOINT_NAME -y
  echo "deleting model..."
  az ml model delete -n tfserving-mounted --version 1
  # </delete_endpoint_and_model>
  cleanup
  exit 1
fi

# Test remotely
echo "Testing endpoint"
for i in {1..10}
do
   # <invoke_endpoint>
   RESPONSE=$(az ml endpoint invoke -n $ENDPOINT_NAME --request-file $BASE_PATH/sample_request.json)
   # </invoke_endpoint>
done

echo "Tested successfully, response was $RESPONSE. Cleaning up..."

cleanup
