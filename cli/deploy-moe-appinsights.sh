#!/bin/bash

# <set_variables> 
RAND="$RANDOM"
ENDPOINT_NAME="endpt-$RAND"
# The fully qualified resource ID ("/subscription/...") of the Application Insights resource to use for logging. 
# If not specified, the default Application Insights resource associated with the workspace will be used.
ALTERNATIVE_APP_INSIGHTS_ID="<ALTERNATIVE_APP_INSIGHTS_ID>"
# </set_variables> 

# <get_details> 
APP_INSIGHTS_RESOURCE_ID=$(az ml workspace show -o tsv --query 'application_insights')
APP_INSIGHTS_INSTRUMENTATION_KEY=$(az monitor app-insights component show --ids $APP_INSIGHTS_ID --query "instrumentationKey" -o tsv)
# </get_details> 

# <create_endpoint> 
az ml online-endpoint create --name $ENDPOINT_NAME
# </create_endpoint>

APP_INSIGHTS_KEY="8d5839d8-f8eb-4ddf-96c1-f06b36ea59c8"

# <create_deployment> 
az ml online-deployment update \
    -f endpoints/online/managed/app-insights/deployment.yml \
    --endpoint $ENDPOINT_NAME \
    --set environment_variables.AML_APP_INSIGHTS_KEY=$APP_INSIGHTS_KEY \
    --all-traffic
# </create_deployment>

# <send_request_1> 
az ml online-endpoint invoke -n $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json
# </send_request_1> 

# <a

# <create_deployment> 
az ml online-deployment update \
    -f endpoints/online/managed/app-insights/deployment.yml \
    --endpoint $ENDPOINT_NAME \
    --set environment_variables.AML_APP_INSIGHTS_KEY=$APP_INSIGHTS_KEY \
    --all-traffic
# </create_deployment>

az ml online-deployment update \
    -f endpoints/online/managed/app-insights/deployment.yml \
    --endpoint $ENDPOINT_NAME \
    --set environment_variables.AML_APP_INSIGHTS_KEY=$APP_INSIGHTS_KEY \
    --set environment_variables.APP_INSIGHTS_LOG_RESPONSE_ENABLED=true \
    --set name="dep1" \
    --all-traffic

az ml online-endpoint invoke -n $ENDPOINT_NAME --request-file endpoints/online/model-1/sample-request.json

az monitor app-insights query -q "requests | where timestamp > ago(1h) | summarize count() by name | order by count_ desc"