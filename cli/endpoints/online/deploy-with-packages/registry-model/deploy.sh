#<get_model>
MODEL_NAME="t5-base"
MODEL_VERSION=$(az ml model show --name $MODEL_NAME --label latest --registry-name azureml | jq .version -r)
#</get_model>

#<build_package>
az ml model package --name $MODEL_NAME \
                    --version $MODEL_VERSION \
                    --registry-name azureml \
                    --file package.yml
#</build_package>