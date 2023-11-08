#<register_model>
MODEL_NAME='sklearn-regression'
MODEL_PATH='model'
az ml model create --name $MODEL_NAME --path $MODEL_PATH --type custom_model
#</register_model>

#<base_environment>
az ml environment create -f environment/sklearn-regression-env.yml
#</base_environment>

#<build_package>
az ml model package -n $MODEL_NAME -l latest --file package-moe.yml
#</build_package>

#<endpoint_name>
ENDPOINT_NAME = "sklearn-regression-online"
#</endpoint_name>

# The following code ensures the created deployment has a unique name
ENDPOINT_SUFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-5} | head -n 1)
ENDPOINT_NAME="$ENDPOINT_NAME-$ENDPOINT_SUFIX"

#<create_endpoint>
az ml online-endpoint create -n $ENDPOINT_NAME
#</create_endpoint>

#<create_deployment>
az ml online-deployment create -f deployment.yml
#</create_deployment>

#<test_deployment>
az ml online-endpoint invoke -n $ENDPOINT_NAME -d with-package -f sample-request.json
#</test_deployment>

#<create_deployment_with_package>
az ml online-deployment create -f model-deployment.yml --with-package
#</create_deployment_with_package>

#<delete_resources>
az ml online-endpoint delete -n $ENDPOINT_NAME --yes
#</delete_resources>

#<build_package_copy>
az ml model package -n $MODEL_NAME -l latest --file package-external.yml
#</build_package_copy>