#!/bin/bash

## IMPORTANT: this file and accompanying assets are the source for snippets in https://docs.microsoft.com/azure/machine-learning! 
## Please reach out to the Azure ML docs & samples team before before editing for the first time.

# <installation>
# az extension add -n ml
# </installation>

# <create environment variables>
export SUBSCRIPTION_ID="5f08d643-1910-4a38-a7c7-84a39d4f42e0"
export RESOURCE_GROUP="trmccorm"
export WORKSPACE="trmccorm-centraluseuap"
export API_VERSION="2021-03-01-preview"
export TOKEN=$(az account get-access-token | jq -r ".accessToken")
#</create environment variables>

# <create resource group>
# az group create -n azureml-examples-cli -l eastus
# </create resource group>

# <create workspace>
# az ml workspace create --name main -g azureml-examples-cli
# </create workspace>

# <configure-defaults>
az configure --defaults workspace=$WORKSPACE
az configure --defaults location="centraluseuap"
az configure --defaults group=$RESOURCE_GROUP
# </configure-defaults>

#TODO upload score file

# <create code>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/score-sklearn/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
  \"properties\": {
    \"description\": \"Score code\",
    \"datastoreId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore\",
    \"path\": \"score\"
  }
}"
# </create code>

# TODO upload model file to blob storage

# <create model>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/sklearn/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer " \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"datastoreId\":\"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore\",
        \"path\": \"model/sklearn_regression_model.pkl\",
        \"description\": \"\",
        \"tags\": {},
        \"properties\": {}
    }
}"
# </create model>

# <create environment>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/sklearn-env/versions/1?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\":{
        \"condaFile\": \"channels:\n  - conda-forge\ndependencies:\n  - python=3.6.1\n  - numpy\n  - pip\n  - scikit-learn==0.19.1\n  - scipy\n  - pip:\n    - azureml-defaults\n    - inference-schema[numpy-support]\n    - joblib\n    - numpy\n    - scikit-learn==0.19.1\n    - scipy\",
        \"Docker\": {
            \"type\": \"Image\",
            \"DockerImageUri\": \"mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20210301.v1\"
        }
    }
}"
# </create environment>

#<create endpoint>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/my-endpoint?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Im5PbzNaRHJPRFhFSzFqS1doWHNsSFJfS1hFZyIsImtpZCI6Im5PbzNaRHJPRFhFSzFqS1doWHNsSFJfS1hFZyJ9.eyJhdWQiOiJodHRwczovL21hbmFnZW1lbnQuY29yZS53aW5kb3dzLm5ldC8iLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC83MmY5ODhiZi04NmYxLTQxYWYtOTFhYi0yZDdjZDAxMWRiNDcvIiwiaWF0IjoxNjE4NDM5OTU1LCJuYmYiOjE2MTg0Mzk5NTUsImV4cCI6MTYxODQ0Mzg1NSwiX2NsYWltX25hbWVzIjp7Imdyb3VwcyI6InNyYzEifSwiX2NsYWltX3NvdXJjZXMiOnsic3JjMSI6eyJlbmRwb2ludCI6Imh0dHBzOi8vZ3JhcGgud2luZG93cy5uZXQvNzJmOTg4YmYtODZmMS00MWFmLTkxYWItMmQ3Y2QwMTFkYjQ3L3VzZXJzLzQ0NGJkNWYyLTJlZGMtNDAzOS1hMDNiLTQwM2FhOGEyMDBhMS9nZXRNZW1iZXJPYmplY3RzIn19LCJhY3IiOiIxIiwiYWlvIjoiQVVRQXUvOFRBQUFBaDIzVmZ5b3JJYVJUOGZ0THI3Z211bnpCRm5ad04xY0VWNE1yYThwUCtMZ2NCT0FWN0RXZmxpejZobmFZcEQ1Sm92VFBVSTUwaXd3WXY0MmlRRyt1V1E9PSIsImFtciI6WyJyc2EiLCJtZmEiXSwiYXBwaWQiOiIwNGIwNzc5NS04ZGRiLTQ2MWEtYmJlZS0wMmY5ZTFiZjdiNDYiLCJhcHBpZGFjciI6IjAiLCJkZXZpY2VpZCI6IjE1YWNhMjhhLTgzMmQtNDg2MS1hYTdlLTFiNzlhMzA2YWY1ZSIsImZhbWlseV9uYW1lIjoiTWNDb3JtaWNrIiwiZ2l2ZW5fbmFtZSI6IlRyZW50IiwiaXBhZGRyIjoiNzMuMTA5LjYxLjM3IiwibmFtZSI6IlRyZW50IE1jQ29ybWljayIsIm9pZCI6IjQ0NGJkNWYyLTJlZGMtNDAzOS1hMDNiLTQwM2FhOGEyMDBhMSIsIm9ucHJlbV9zaWQiOiJTLTEtNS0yMS0yMTI3NTIxMTg0LTE2MDQwMTI5MjAtMTg4NzkyNzUyNy0yNDgwNTIxMSIsInB1aWQiOiIxMDAzQkZGRDlENjA1RUVGIiwicmgiOiIwLkFSb0F2NGo1Y3ZHR3IwR1JxeTE4MEJIYlI1VjNzQVRialJwR3UtNEMtZUdfZTBZYUFDVS4iLCJzY3AiOiJ1c2VyX2ltcGVyc29uYXRpb24iLCJzdWIiOiJrbkp6c0NYUVp0cjhWY01vSy16VzZwdG5tVzdqOFlYQU43QnU0RnpwUnlrIiwidGlkIjoiNzJmOTg4YmYtODZmMS00MWFmLTkxYWItMmQ3Y2QwMTFkYjQ3IiwidW5pcXVlX25hbWUiOiJ0cm1jY29ybUBtaWNyb3NvZnQuY29tIiwidXBuIjoidHJtY2Nvcm1AbWljcm9zb2Z0LmNvbSIsInV0aSI6IlpmeU1mRG9sR0Vpb3p3YU9WOGNKQUEiLCJ2ZXIiOiIxLjAiLCJ3aWRzIjpbImI3OWZiZjRkLTNlZjktNDY4OS04MTQzLTc2YjE5NGU4NTUwOSJdLCJ4bXNfdGNkdCI6MTI4OTI0MTU0N30.RR8PDEXb7LNrUEhSpFsb2ojmpF5RmsmkLTZSusOQczY_KoAnET3QsdWhQ8s_iFB9TFYPmx-uDpvB5eA0WhXmK_ectOqTwGLoZPtbPGjMnDtX3ds8gir4tMuW4pjWVCjXaM8yc_T8w667bjdIrXuASst4M9DZLjsnuqGsXRW38NHAHePXrjPiyQ7WPalSQ0CphwBmw3Z0oC16vywYsixw3nqX-LyMJmqHL-zfkdPOjoUHIrqlOOmBkWH7wyKnYRX91uywkn01G1vTkK0aOoOcOSjNZLFUvnaS92L30OOe5GDUAcQ3O1OAq6zUhrVle7mYBucmnNXqS5W8oKaJTr3tcQ" \
--data-raw "{
    \"properties\": {
        \"authMode\": \"AMLToken\",
        \"traffic\": { \"blue\": 100 }
    },
    \"location\": \"westus\"
}"
#</create endpoint>

# todo: missing discriminator bug
#<create deployment>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/my-endpoint1/deployments/blue?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Im5PbzNaRHJPRFhFSzFqS1doWHNsSFJfS1hFZyIsImtpZCI6Im5PbzNaRHJPRFhFSzFqS1doWHNsSFJfS1hFZyJ9.eyJhdWQiOiJodHRwczovL21hbmFnZW1lbnQuY29yZS53aW5kb3dzLm5ldC8iLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC83MmY5ODhiZi04NmYxLTQxYWYtOTFhYi0yZDdjZDAxMWRiNDcvIiwiaWF0IjoxNjE4NDM5OTU1LCJuYmYiOjE2MTg0Mzk5NTUsImV4cCI6MTYxODQ0Mzg1NSwiX2NsYWltX25hbWVzIjp7Imdyb3VwcyI6InNyYzEifSwiX2NsYWltX3NvdXJjZXMiOnsic3JjMSI6eyJlbmRwb2ludCI6Imh0dHBzOi8vZ3JhcGgud2luZG93cy5uZXQvNzJmOTg4YmYtODZmMS00MWFmLTkxYWItMmQ3Y2QwMTFkYjQ3L3VzZXJzLzQ0NGJkNWYyLTJlZGMtNDAzOS1hMDNiLTQwM2FhOGEyMDBhMS9nZXRNZW1iZXJPYmplY3RzIn19LCJhY3IiOiIxIiwiYWlvIjoiQVVRQXUvOFRBQUFBaDIzVmZ5b3JJYVJUOGZ0THI3Z211bnpCRm5ad04xY0VWNE1yYThwUCtMZ2NCT0FWN0RXZmxpejZobmFZcEQ1Sm92VFBVSTUwaXd3WXY0MmlRRyt1V1E9PSIsImFtciI6WyJyc2EiLCJtZmEiXSwiYXBwaWQiOiIwNGIwNzc5NS04ZGRiLTQ2MWEtYmJlZS0wMmY5ZTFiZjdiNDYiLCJhcHBpZGFjciI6IjAiLCJkZXZpY2VpZCI6IjE1YWNhMjhhLTgzMmQtNDg2MS1hYTdlLTFiNzlhMzA2YWY1ZSIsImZhbWlseV9uYW1lIjoiTWNDb3JtaWNrIiwiZ2l2ZW5fbmFtZSI6IlRyZW50IiwiaXBhZGRyIjoiNzMuMTA5LjYxLjM3IiwibmFtZSI6IlRyZW50IE1jQ29ybWljayIsIm9pZCI6IjQ0NGJkNWYyLTJlZGMtNDAzOS1hMDNiLTQwM2FhOGEyMDBhMSIsIm9ucHJlbV9zaWQiOiJTLTEtNS0yMS0yMTI3NTIxMTg0LTE2MDQwMTI5MjAtMTg4NzkyNzUyNy0yNDgwNTIxMSIsInB1aWQiOiIxMDAzQkZGRDlENjA1RUVGIiwicmgiOiIwLkFSb0F2NGo1Y3ZHR3IwR1JxeTE4MEJIYlI1VjNzQVRialJwR3UtNEMtZUdfZTBZYUFDVS4iLCJzY3AiOiJ1c2VyX2ltcGVyc29uYXRpb24iLCJzdWIiOiJrbkp6c0NYUVp0cjhWY01vSy16VzZwdG5tVzdqOFlYQU43QnU0RnpwUnlrIiwidGlkIjoiNzJmOTg4YmYtODZmMS00MWFmLTkxYWItMmQ3Y2QwMTFkYjQ3IiwidW5pcXVlX25hbWUiOiJ0cm1jY29ybUBtaWNyb3NvZnQuY29tIiwidXBuIjoidHJtY2Nvcm1AbWljcm9zb2Z0LmNvbSIsInV0aSI6IlpmeU1mRG9sR0Vpb3p3YU9WOGNKQUEiLCJ2ZXIiOiIxLjAiLCJ3aWRzIjpbImI3OWZiZjRkLTNlZjktNDY4OS04MTQzLTc2YjE5NGU4NTUwOSJdLCJ4bXNfdGNkdCI6MTI4OTI0MTU0N30.RR8PDEXb7LNrUEhSpFsb2ojmpF5RmsmkLTZSusOQczY_KoAnET3QsdWhQ8s_iFB9TFYPmx-uDpvB5eA0WhXmK_ectOqTwGLoZPtbPGjMnDtX3ds8gir4tMuW4pjWVCjXaM8yc_T8w667bjdIrXuASst4M9DZLjsnuqGsXRW38NHAHePXrjPiyQ7WPalSQ0CphwBmw3Z0oC16vywYsixw3nqX-LyMJmqHL-zfkdPOjoUHIrqlOOmBkWH7wyKnYRX91uywkn01G1vTkK0aOoOcOSjNZLFUvnaS92L30OOe5GDUAcQ3O1OAq6zUhrVle7mYBucmnNXqS5W8oKaJTr3tcQ" \
--data-raw "{
    \"location\": \"centraluseuap\",
    \"properties\": {
        \"type\": \"Managed\",
        \"scaleSettings\": {
            \"scaleType\": \"Manual\",
            \"instanceCount\": 1,
            \"minInstances\": 1,
            \"maxInstances\": 2
        },
        \"model\": {
            \"referenceType\": \"Id\",
            \"id\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/models/sklearn/versions/1\"
        },
        \"codeConfiguration\": {
            \"codeId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/score-sklearn/versions/1\",
            \"scoringScript\": \"score.py\"
        },
        \"environmentId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/sklearn-env\",
        \"InstanceType\": \"Standard_F2s_v2\"
    }
}"
#</create deployment>