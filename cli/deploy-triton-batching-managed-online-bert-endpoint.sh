##!/bin/bash
## Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions
## are met:
##  * Redistributions of source code must retain the above copyright
##    notice, this list of conditions and the following disclaimer.
##  * Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  * Neither the name of NVIDIA CORPORATION nor the names of its
##    contributors may be used to endorse or promote products derived
##    from this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
## EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
## OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



set -e

BASE_PATH=endpoints/online/triton/single-model
MODEL_PATH=$BASE_PATH/models/triton/bert-si-onnx/1

# <set_endpoint_name>
export ENDPOINT_NAME=triton-bert-endpt
# </set_endpoint_name>

# Download the model
mkdir -p $MODEL_PATH
wget -O $MODEL_PATH/model.onnx https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/model/bertsquad-8.onnx?raw=true 

#Download the dependencies file required by BERT script (tokenization script and helper functions)
wget -O $BASE_PATH/run_onnx_squad.py https://raw.githubusercontent.com/onnx/models/master/text/machine_comprehension/bert-squad/dependencies/run_onnx_squad.py
wget -O $BASE_PATH/tokenization.py https://raw.githubusercontent.com/onnx/models/master/text/machine_comprehension/bert-squad/dependencies/tokenization.py
wget -O $BASE_PATH/uncased_L-12_H-768_A-12.zip -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip 
unzip $BASE_PATH/uncased_L-12_H-768_A-12.zip  -d $BASE_PATH/ && rm $BASE_PATH/uncased_L-12_H-768_A-12.zip 

# Generate the sample input file to be used by the scoring script later
# python3 $BASE_PATH/generate_input_data.py

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

# <get_logs>
az ml endpoint get-logs -n $ENDPOINT_NAME --deployment blue
# </get_logs>


# <get_scoring_uri>
scoring_uri=$(az ml endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
scoring_uri=${scoring_uri%/*} # this will get rid of score
scoring_uri=${scoring_uri//triton/blue-triton} # this will add the 'blue' as this is the bug which needs deployment name (blue in this case)
# </get_scoring_uri>

# <get_token>
auth_token=$(az ml endpoint get-credentials -n $ENDPOINT_NAME --query accessToken -o tsv)
# </get_token>

# < get the primary key>
primary_key =$(az ml endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey o tsv)
# </ get the primary key>

# <check_scoring_of_model>
python3 $BASE_PATH/triton_bert_scoring.py --base_url=$scoring_uri --token=$auth_token --model="bert-squad" # the default model is bert-squad-batch which uses optimized config
# </check_scoring_of_model>

# <set filename to store perf_analyzer outputs>
export filename=bert_squad.csv
# < Run the perf-analyzer for the BERT default onnx model>
perf_analyzer -m bert-squad -b 1 --concurrency-range 2:64:2 -u $scoring_uri -H 'Authorization: Bearer '$primary_key -f $filename
column -s, -t < $filename | less -#2 -N -S

# </check perf-analyzer>

#run perf_analyzer with the optimized config
export filename=bert_squad_batch.csv
perf_analyzer -m bert-squad-bath -b 1 --concurrency-range 2:64:2 -u $scoring_uri -H 'Authorization: Bearer '$primary_key -f $filename
column -s, -t < $filename | less -#2 -N -S

#</run perf optimized config>

# <delete_endpoint>
az ml endpoint delete -n $ENDPOINT_NAME --yes
# </delete_endpoint>

# <delete_model>
az ml model delete -n bert-si-onnx --version 4
# </delete_model>


