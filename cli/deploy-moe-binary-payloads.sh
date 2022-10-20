# <set_variables> 
ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)
# </set_variables>

BASE_PATH="endpoints/online/managed/binary-payloads"

# Helper function to change parameters in yaml files
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

cleanup_() {
    rm peacock-pic.jpg
    rm out-1.jpg
}

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
  #exit 1
fi

# <create_deployment_1>
SCORING_SCRIPT="single-file-to-file-score.py"
change_vars $BASE_PATH/binary-payloads-deployment.yml
az ml online-deployment create -e $ENDPOINT_NAME -f $BASE_PATH/binary-payloads-deployment.yml_ --all-traffic 
# </create_deployment_1> 

az ml online-deployment get-logs -n binary-payload -e $ENDPOINT_NAME 

# Check if deployment was successful 
deploy_status=`az ml online-deployment show --name minimal-multimodel --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv `
echo $deploy_status
if [[ $deploy_status == "Succeeded" ]]
then
  echo "Deployment completed successfully"
else
  echo "Deployment failed"
  #exit 1
fi

# Get key
echo "Getting access key..."
KEY=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query primaryKey -o tsv )

# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv )
echo "Scoring url is $SCORING_URL"

# Get test image
wget https://aka.ms/peacock-pic -O peacock-pic.jpg

# <test_online_endpoint_1> 
curl -X POST -F "file=@peacock-pic.jpg" -H "Authorization: Bearer $KEY"   $SCORING_URL -o $BASE_PATH/out-1.jpg
# <test_online_endpoint_1> 

# <create_deployment_2>
SCORING_SCRIPT="multi-file-to-json-score.py"
change_vars $BASE_PATH/binary-payloads-deployment.yml
az ml online-deployment create -e $ENDPOINT_NAME -f $BASE_PATH/binary-payloads-deployment.yml_ --all-traffic 
az ml online-deployment update -e $ENDPOINT_NAME -f $BASE_PATH/binary-payloads-deployment.yml_ 
# </create_deployment_2> 

# <test_online_endpoint_2>
#curl -X POST -F "file1=@peacock-pic.jpg" -F "file2=@$BASE_PATH/out-1.jpg" -H "Authorization: Bearer $KEY"  $SCORING_URL
curl -X POST -F "file[]=@peacock-pic.jpg" -F "file[]=@$BASE_PATH/out-1.jpg" -H "Authorization: Bearer $KEY"  $SCORING_URL
# <test_online_endpoint_2> 