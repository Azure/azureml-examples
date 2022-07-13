#!/bin/bash

set -e

pip install pandas numpy scikit-learn joblib

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<ACR_NAME>"
export ACR_GROUP="<ACR_GROUP>"
# </set_variables> 

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)

# <set_base_path_and_build_dirs> 
export BASE_PATH="endpoints/online/custom-container/multimodel-champion-challenger"
mkdir -p $BASE_PATH/{models,test-data,build,code}
PARENT_PATH=$(dirname $BASE_PATH)
cp $PARENT_PATH/{mm-cc-train.py,mm-cc-deployment.yml,mm-cc-endpoint.yml} $BASE_PATH/ 
cp $PARENT_PATH/mm-cc.dockerfile $BASE_PATH/build/
cp $PARENT_PATH/mm-cc-score.py $BASE_PATH/code/
# </set_base_path_and_build_dirs> 

cd $BASE_PATH

# <train_models_and_generate_test_data>
python mm-cc-train.py
# </train_models_and_generate_test_data> 

cd - 

# <create_endpoint>
az ml online-endpoint create -f $BASE_PATH/mm-cc-endpoint.yml -e $ENDPOINT_NAME
# </create_endpoint>

endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`

echo $endpoint_status

if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else 
  echo "Endpoint creation failed"
  exit 1
fi

# <login_to_acr> 
az acr login -n $ACR_NAME
# </login_to_acr> 

# <build_image> 
az acr build -t championchallengercc:1 -r $ACR_NAME --resource-group $ACR_GROUP -f $BASE_PATH/build/mm-cc.dockerfile $BASE_PATH/build 
# </build_image> 

cp $BASE_PATH/mm-cc-deployment.yml $BASE_PATH/mm-cc-deployment_.yml
sed -i "s/ACR_NAME/$ACR_NAME/g;" $BASE_PATH/mm-cc-deployment.yml

# <create_deployment>
az ml online-deployment create -f $BASE_PATH/mm-cc-deployment.yml -e $ENDPOINT_NAME --all-traffic 
# </create_deployment>

mv $BASE_PATH/mm-cc-deployment_.yml $BASE_PATH/mm-cc-deployment.yml

deploy_status=`az ml online-deployment show --name championchallengercc --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
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

# <test_online_endpoint>
curl -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" -d @$BASE_PATH/test-data/test-data-1.json $SCORING_URL
# </test_online_endpoint> 

# <delete_endpoint>
az ml online-endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint>
