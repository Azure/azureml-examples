BASE_PATH=endpoints/online/custom-container
wget https://aka.ms/half_plus_two-model -O $BASE_PATH/half_plus_two.tar.gz
tar -xvf $BASE_PATH/half_plus_two.tar.gz -C $BASE_PATH

# GET ACR
ACR_NAME=$( az ml workspace show -n gopalv-test -g gopalv-arm-centraluseuap --query container_registry)
docker build $BASE_PATH -f ./tfserving.dockerfile -t $ACR_NAME.azurecr.io/tf-serving:8501-env-variables-mount
docker push $ACR_NAME.azurecr.io/tf-serving:8501-env-variables-mount
az ml endpoint create -f TFServing-endpoint.yaml

rm $BASE_PATH/half_plus_two.tar.gz
rm -r $BASE_PATH/half_plus_two