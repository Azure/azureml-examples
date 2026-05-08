# <initialize_variables>
BASE_PATH=endpoints/online/mlnet
ENDPOINT_NAME=mlnet-endpoint
DEPLOYMENT_NAME=mlnet
ENVIRONMENT_NAME=mlnet-environment
# </initialize_variables>

# <run_image_locally_for_testing>
docker run -d --rm -p 5000:80 --name="mlnet-test" docker.io/luisquintanilla/managed-deployment-container-mlnet:latest
sleep 10
# </run_image_locally_for_testing>

# <check_liveness_locally>
curl -v http://localhost:5000
# </check_liveness_locally>

# <check_scoring_locally>
curl --header "Content-Type: application/json" \
--request POST --data @$BASE_PATH/sample_request.json \
http://localhost:5000/predict
# </check_scoring_locally>

# <stop_image>
docker stop mlnet-test
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
  # <delete_endpoint_and_environment>
  az ml endpoint delete -n $ENDPOINT_NAME -y
  echo "Deleting environment"
  az ml environment delete --name $ENVIRONMENT_NAME
  # </delete_endpoint_and_environment>
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