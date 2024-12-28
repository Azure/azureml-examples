# Note: this is based on the deploy-managed-online-endpoint.sh: it just adds autoscale settings in the end
set -e

#set the endpoint name from the how-to-deploy excercise
# <set_endpoint_deployment_name>
# set your existing endpoint name
ENDPOINT_NAME=your-endpoint-name
DEPLOYMENT_NAME=blue
# </set_endpoint_deployment_name>

export ENDPOINT_NAME=autoscale-endpt-`echo $RANDOM`

# create endpoint and deployment
az ml online-endpoint create --name $ENDPOINT_NAME -f endpoints/online/managed/sample/endpoint.yml
az ml online-deployment create --name blue --endpoint $ENDPOINT_NAME -f endpoints/online/managed/sample/blue-deployment.yml --all-traffic
az ml online-endpoint show -n $ENDPOINT_NAME


# <set_other_env_variables>
# ARM id of the deployment
DEPLOYMENT_RESOURCE_ID=$(az ml online-deployment show -e $ENDPOINT_NAME -n $DEPLOYMENT_NAME -o tsv --query "id")
# ARM id of the deployment. todo: change to --query "id"
ENDPOINT_RESOURCE_ID=$(az ml online-endpoint show -n $ENDPOINT_NAME -o tsv --query "properties.\"azureml.onlineendpointid\"")
# set a unique name for autoscale settings for this deployment. The below will append a random number to make the name unique.
AUTOSCALE_SETTINGS_NAME=autoscale-$ENDPOINT_NAME-$DEPLOYMENT_NAME-`echo $RANDOM`
# </set_other_env_variables>

# create autoscale settings. Note if you followed the how-to-deploy doc example, the instance count would have been 1. Now after applying this poilcy, it will scale up 2 (since min count and count are 2).
# <create_autoscale_profile>
az monitor autoscale create \
  --name $AUTOSCALE_SETTINGS_NAME \
  --resource $DEPLOYMENT_RESOURCE_ID \
  --min-count 2 --max-count 5 --count 2
# </create_autoscale_profile>

# Add rule to default profile: scale up if cpu util > 70 %
# <scale_out_on_cpu_util>
az monitor autoscale rule create \
  --autoscale-name $AUTOSCALE_SETTINGS_NAME \
  --condition "CpuUtilizationPercentage > 70 avg 5m" \
  --scale out 2
# </scale_out_on_cpu_util>

# Add rule to default profile: scale down if cpu util < 25 %
# <scale_in_on_cpu_util>
az monitor autoscale rule create \
  --autoscale-name $AUTOSCALE_SETTINGS_NAME \
  --condition "CpuUtilizationPercentage < 25 avg 5m" \
  --scale in 1
# </scale_in_on_cpu_util>

# add rule to default profile: scale up based on avg. request latency (endpoint metric)
# <scale_up_on_request_latency>
az monitor autoscale rule create \
 --autoscale-name $AUTOSCALE_SETTINGS_NAME \
 --condition "RequestLatency > 70 avg 5m" \
 --scale out 1 \
 --resource $ENDPOINT_RESOURCE_ID
# </scale_up_on_request_latency>

# create weekend profile: scale to 2 nodes in weekend
# <weekend_profile>
az monitor autoscale profile create \
  --name weekend-profile \
  --autoscale-name $AUTOSCALE_SETTINGS_NAME \
  --min-count 2 --count 2 --max-count 2 \
  --recurrence week sat sun --timezone "Pacific Standard Time" 
# </weekend_profile>

# disable the autoscale profile
# <disable_profile>
az monitor autoscale update \
  --autoscale-name $AUTOSCALE_SETTINGS_NAME \
  --enabled false
# </disable_profile>

# <delete_endpoint>
# delete the autoscaling profile
az monitor autoscale delete -n "$AUTOSCALE_SETTINGS_NAME"

# delete the endpoint
az ml online-endpoint delete --name $ENDPOINT_NAME --yes --no-wait
# </delete_endpoint>
