#!/bin/bash

set -e 

# <set_parameters> 
ENDPOINT_NAME=hf-tg-`echo $RANDOM`
IMAGE_TAG=azureml-examples/huggingface-textgen:1 

BASE_PATH=endpoints/online/custom-container/torchserve/huggingface-textgen
SERVE_PATH=$BASE_PATH/serve/examples/Huggingface_Transformers
ROOT_PATH=$PWD
ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)
# </set_parameters> 

# <define_helper_functions> 
# Helper function to parameterize YAML
change_vars() {
  for FILE in "$@"; do 
    TMP="${FILE}_"
    cp $FILE $TMP 
    readarray -t VARS < <(cat $TMP | grep -oP '{{.*?}}' | sed -e 's/[}{]//g'); 
    for VAR in "${VARS[@]}"; do
      sed -i "s#{{${VAR}}}#${!VAR}#g" $TMP
    done
  done
}

cleanup () {
    sudo rm -rf $BASE_PATH/serve || true
    rm $BASE_PATH/ts-hf-tg-deployment.yml_
    az ml online-endpoint delete -y -n $ENDPOINT_NAME
}
# </define_helper_functions> 

# <build_image>
az acr login -n $ACR_NAME
az acr build -t $IMAGE_TAG -f $BASE_PATH/ts-hf-tg.dockerfile -r $ACR_NAME $BASE_PATH
# </build_image> 

# <download_huggingface_assets> 
cd $BASE_PATH
rm -rf serve
git init serve
cd serve
git remote add -f origin https://github.com/pytorch/serve
git config core.sparseCheckout true
echo "examples/Huggingface_Transformers" >> .git/info/sparse-checkout
git pull origin master
cd $ROOT_PATH
# </download_huggingface_assets> 

# <generate_model>
if [[ ! -f $SERVE_PATH/setup_config.json_ ]]; then
    cp $BASE_PATH/ts-hf-tg-setup_config.json $SERVE_PATH/setup_config.json_
fi
cp $BASE_PATH/ts-hf-tg-setup_config.json $SERVE_PATH/setup_config.json
chmod -R o+w $SERVE_PATH
cd $SERVE_PATH
docker run --rm -v $PWD:/tmp/wd:z -w /tmp/wd -t "$ACR_NAME.azurecr.io/$IMAGE_TAG" "python Download_Transformer_models.py; \
    sed -i 's#\"max_length\": 50#\"max_length\": 300#g' ./Transformer_model/config.json; \
    torch-model-archiver --model-name Textgeneration --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files Transformer_model/config.json,./setup_config.json; \
    exit;"
cp setup_config.json_ setup_config.json
rm setup_config.json_
cd $ROOT_PATH
# </generate_model>

# <test_locally> 
docker run --name huggingface-textgen --rm -d -p 8080:8080 -v "$PWD/$SERVE_PATH":/tmp/wd -e AZUREML_MODEL_DIR=/tmp/wd -e TORCHSERVE_MODELS="textgeneration=Textgeneration.mar" -t "$ACR_NAME.azurecr.io/$IMAGE_TAG" 
sleep 10
curl -X POST http://127.0.0.1:8080/predictions/textgeneration -T "$SERVE_PATH/Text_gen_artifacts/sample_text.txt"
docker stop huggingface-textgen
# </test_locally> 

# <create_endpoint> 
az ml online-endpoint create -n $ENDPOINT_NAME
# </create_endpoint> 

# <check_endpoint_status> 
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else 
  echo "Endpoint creation failed"
  exit 1
fi
# </check_endpoint_status> 

# <create_deployment> 
change_vars $BASE_PATH/ts-hf-tg-deployment.yml
az ml online-deployment create -f $BASE_PATH/ts-hf-tg-deployment.yml_ --all-traffic
# </create_deployment> 

# <check_deployment_status> 
deploy_status=`az ml online-deployment show --endpoint-name $ENDPOINT_NAME --name textgeneration --query "provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
    echo "Deployment completed successfully"
else
    echo "Deployment failed"
    exit 1
fi
# </check_deployment_status> 

# <get_endpoint_details> 
# Get key
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv)

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"
# </get_endpoint_details> 

# <test_endpoint> 
curl -X POST -H "Authorization: Bearer $KEY" -T "$SERVE_PATH/Text_gen_artifacts/sample_text.txt" $SCORING_URL
# </test_endpoint> 

cleanup