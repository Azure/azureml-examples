#!/bin/bash 
set -e

# <set_variables> 
ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)
# </set_variables>

BASE_PATH="endpoints/online/managed/binary-payloads"

# <download_sample_data> 
wget https://aka.ms/peacock-pic -O endpoints/online/managed/binary-payloads/input.jpg
# </download_sample_data>

# <create_endpoint> 
az ml online-endpoint create -n $ENDPOINT_NAME  
# </create_endpoint> 

# Check if endpoint was successful
endpoint_status=`az ml online-endpoint show --name $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $endpoint_status
if [[ $endpoint_status == "Succeeded" ]]
then
  echo "Endpoint created successfully"
else
  echo "Endpoint creation failed"
  exit 1
fi

# <create_deployment>
az ml online-deployment create -e $ENDPOINT_NAME -f $BASE_PATH/binary-payloads-deployment.yml \
  --set code_configuration.scoring_script=single-file-to-file-score.py \
  --all-traffic 
# </create_deployment>

# <get_endpoint_details> 
# Get key
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv )

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv )
echo "Scoring url is $SCORING_URL"
# </get_endpoint_details> 

# <get_logs> 
az ml online-deployment get-logs -n binary-payload -e $ENDPOINT_NAME 
# </get_logs> 

# <check_deployment> 
# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name binary-payload --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  exit 1
fi
# </check_deployment> 

# <test_online_endpoint_1> 
curl -X POST -F "file=@endpoints/online/managed/binary-payloads/input.jpg" -H "Authorization: Bearer $KEY"   $SCORING_URL \
  -o endpoints/online/managed/binary-payloads/binary-payloads/output.jpg
# <test_online_endpoint_1> 

# <update_deployment2>
az ml online-deployment update -e $ENDPOINT_NAME -n binary-payload \
  --set code_configuration.scoring_script="multi-file-to-json-score.py" 
# </updat _deployment2> 

# <test_online_endpoint_2>
curl -X POST -F "file[]=@endpoints/online/managed/binary-payloads/input.jpg" \
  -F "file[]=@endpoints/online/managed/binary-payloads/output.jpg" \
  -H "Authorization: Bearer $KEY"  $SCORING_URL
# <test_online_endpoint_2> 

# <delete_assets>
az ml online-endpoint delete -n $ENDPOINT_NAME --no-wait --yes 
# </delete_assets> 