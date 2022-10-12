#!/bin/bash

BASE_PATH=endpoints/online/custom-container/tfserving/half-plus-two-integrated
ENDPOINT_NAME=tfsintegrated-`echo $RANDOM`
ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)

# Helper function to parameterize YAML
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

# <build_image> 
az acr login -n $ACR_NAME
az acr build -f $BASE_PATH/half-plus-two-integrated.Dockerfile -t azureml-examples/tfsintegrated:1 -r $ACR_NAME $BASE_PATH
# </build_image> 

# <test_locally>
docker run -p 8501:8501 --rm --name tfsintegrated -d -t "$ACR_NAME.azurecr.io/azureml-examples/tfsintegrated:1"
sleep 5
curl -d '{"inputs": [[1,1]]}' -H "Content-Type: application/json" localhost:8501/v1/models/hpt:predict
docker stop tfsintegrated
# </test_locally>

# <create_endpoint>
az ml online-endpoint create -n $ENDPOINT_NAME
# </create_endpoint>

# <get_endpoint_details> 
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
# </get_endpoint_details> 

# <create_deployment>
change_vars $BASE_PATH/half-plus-two-integrated-deployment.yml
az ml online-deployment create -f $BASE_PATH/half-plus-two-integrated-deployment.yml_ --all-traffic
rm $BASE_PATH/half-plus-two-integrated-deployment.yml_deployment.yml_
# </create_deployment>

# <test_deployment>
curl -d @$BASE_PATH/sample-data.json -H "Content-Type: application/json" -H "Authorization: Bearer $KEY" $SCORING_URL
# </test_deployment>

# <delete_online_endpoint>
az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>
