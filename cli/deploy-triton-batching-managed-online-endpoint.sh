## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

set -e

BASE_PATH=endpoints/online/triton/batching
DEFAULT_MODEL_PATH=$BASE_PATH/models/triton/bert-squad/1
BATCH_MODEL_PATH=$BASE_PATH/models/triton/bert-squad-batch/1

# <set_endpoint_name>
export ENDPOINT_NAME=triton-batch-endpt-`echo $RANDOM`
# </set_endpoint_name>

# Download the model
mkdir -p $DEFAULT_MODEL_PATH
wget -O $DEFAULT_MODEL_PATH/model.onnx https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/model/bertsquad-8.onnx?raw=true 
mkdir -p $BATCH_MODEL_PATH
cp $DEFAULT_MODEL_PATH/model.onnx $BATCH_MODEL_PATH

#Download the dependencies file required by BERT script (tokenization script and helper functions)
wget -O $BASE_PATH/run_onnx_squad.py https://raw.githubusercontent.com/onnx/models/master/text/machine_comprehension/bert-squad/dependencies/run_onnx_squad.py
wget -O $BASE_PATH/tokenization.py https://raw.githubusercontent.com/onnx/models/master/text/machine_comprehension/bert-squad/dependencies/tokenization.py
wget -O $BASE_PATH/uncased_L-12_H-768_A-12.zip -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip 
unzip $BASE_PATH/uncased_L-12_H-768_A-12.zip  -d $BASE_PATH/ && rm $BASE_PATH/uncased_L-12_H-768_A-12.zip 
# <deploy>
az ml endpoint create -n $ENDPOINT_NAME -f $BASE_PATH/create-endpoint-with-deployment-mir.yml
# </deploy>

# <get_status>
az ml endpoint show -n $ENDPOINT_NAME
# </get_status>

# check if endpoint create was successful
endpoint_status=`az ml endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then  
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

deploy_status=`az ml endpoint show --name $ENDPOINT_NAME --query "deployments[?name=='blue'].provisioning_state" -o tsv`
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi

# <get_scoring_uri>
scoring_uri=$(az ml endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
scoring_uri=${scoring_uri%/*} # this will get rid of score
scoring_uri=${scoring_uri//triton/blue-triton} # this will add the 'blue' as this is the bug which needs deployment name (blue in this case)
# </get_scoring_uri>

# <get_token>
auth_token=$(az ml endpoint get-credentials -n $ENDPOINT_NAME --query accessToken -o tsv)
# </get_token>

# <check_scoring_of_model>
python $BASE_PATH/triton_bert_scoring.py --base_url=$scoring_uri --token=$auth_token
# </check_scoring_of_model>

# <Run the perf-analyzer for the BERT onnx model>
export filename=bert_cuda_staticbs1_fp32.csv
perf_analyzer -m bert-si-onnx -b 1 --concurrency-range 2:64:2 -u $scoring_uri -H 'Authorization: Bearer '$primary_key -f $filename
# </Run the perf-analyzer for the BERT onnx model>

# <delete_endpoint>
az ml endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint>

# <delete_model>
az ml model delete -n bert --version 1
# </delete_model>
