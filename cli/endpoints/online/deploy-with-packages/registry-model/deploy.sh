#<get_model>
MODEL_NAME="t5-base"
MODEL_VERSION=$(az ml model show --name $MODEL_NAME --label latest --registry-name azureml | jq .version -r)
#</get_model>

#<configure_package_target>
SUBSCRIPTION_ID=$(az account show | jq .tenantId -r)
RESOURCE_GROUP_NAME=$(az configure --list-defaults | jq '.[] | select(.name == "group") | .value' -r)
WORKSPACE_NAME=$(az configure --list-defaults | jq '.[] | select(.name == "workspace") | .value' -r)
MODEL_PACKAGE_NAME="pkg-$MODEL_NAME-$MODEL_VERSION"
MODEL_PACKAGE_VERSION=$(date +%s)

TARGET_ENVIRONMENT="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP_NAME/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE_NAME/environments/$MODEL_PACKAGE_NAME/versions/$MODEL_PACKAGE_VERSION"
#</configure_package_target>

#<build_package>
az ml model package --name $MODEL_NAME \
                    --version $MODEL_VERSION \
                    --registry-name azureml \
                    --file package.yml \
                    --set target_environment_name=$TARGET_ENVIRONMENT
#</build_package>