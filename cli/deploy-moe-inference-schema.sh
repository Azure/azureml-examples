#!/bin/bash

set -e

# <set_variables>
export ENDPOINT_NAME="<ENDPOINT_NAME>"
export ACR_NAME="<CONTAINER_REGISTRY_NAME>"
# </set_variables>

export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)
export BASE_PATH=endpoints/online/custom-container/inference-schema

#TODO: delete
ENDPOINT_NAME=azureml-infschema1
ACR_NAME=valwallaceskr

# <login_to_acr>
az acr login -n ${ACR_NAME} 
# </login_to_acr> 

# TODO: version 
#<build_with_acr> 
az acr build -t azureml-examples/infschema:1 -r $ACR_NAME --file $BASE_PATH/inference-schema.dockerfile $BASE_PATH
# </build_with_acr>

#<get_bert_model_assets>
rm -rf $BASE_PATH/models/bert-base-uncased && mkdir -p $BASE_PATH/models/bert-base-uncased
wget --directory-prefix $BASE_PATH/models/bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
wget --directory-prefix $BASE_PATH/models/bert-base-uncased https://huggingface.co/bert-base-uncased/raw/main/{config,tokenizer,tokenizer_config}.json
#</get_bert_model_assets>

#<create_model> 
az ml model create -f $BASE_PATH/model.yml
#</create_model> 

#<create_endpoint> 
az ml online-endpoint create -n $ENDPOINT_NAME -f $BASE_PATH/endpoint.yml
#</create_endpoint> 

endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`

echo $endpoint_status

if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else 
  echo "Endpoint creation failed"
  exit 1
fi

cp $BASE_PATH/deployment.yml $BASE_PATH/deployment-standard.yml
IMAGE_NAME=$ACR_NAME.azurecr.io/azureml-examples/infschema:1
sed -i "s#{{image_name}}#$IMAGE_NAME#g;" $BASE_PATH/deployment-standard.yml

#<create_standard_deployment>
az ml online-deployment create -e $ENDPOINT_NAME -f $BASE_PATH/deployment-standard.yml --all-traffic
#</create_standard_deployment> 

#<check_deploy_status>
az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name 
#</check_deploy_status>

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

#<test_standard_parameter>
curl -X POST -d @$BASE_PATH/sample-inputs/standard.json -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" $SCORING_URL 
#</test_standard_parameter>

#<update_deployment_to_numpy> 
az ml online-deployment update -e $ENDPOINT_NAME -n infschema --set code_configuration.scoring_script=score-numpy-parameter.py
#<update_deployment_to_numpy> 

#<test_numpy_parameter>
curl -X POST -d @$BASE_PATH/sample-inputs/numpy.json -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" $SCORING_URL 
#</test_numpy_parameter>

#<update_deployment_to_pandas> 
az ml online-deployment update -e $ENDPOINT_NAME -n infschema --set code_configuration.scoring_script=score-pandas-parameter.py
#<update_deployment_to_pandas> 

#<test_pandas_parameter>
curl -X POST -d @$BASE_PATH/sample-inputs/pandas.json -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" $SCORING_URL 
#</test_pandas_parameter>

#<update_deployment_to_nested> 
az ml online-deployment update -e $ENDPOINT_NAME -n infschema --set code_configuration.scoring_script=score-nested-parameter.py
#<update_deployment_to_nested> 

#<test_nested_parameter>
curl -X POST -d @$BASE_PATH/sample-inputs/nested.json -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" $SCORING_URL 
#</test_nested_parameter>

#<update_deployment_to_abstract> 
az ml online-deployment update -e $ENDPOINT_NAME -n infschema --set code_configuration.scoring_script=score-abstract-parameter.py
#<update_deployment_to_abstract> 

#<test_abstract_parameter>
curl -X POST -d @$BASE_PATH/sample-inputs/abstract.json -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" $SCORING_URL 
#</test_abstract_parameter>

# <delete_endpoint>
az ml online-endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint>