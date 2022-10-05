# Deploy a custom container using MLFlow
This example shows how to deploy two MLFlow models that each have different conda environments using a custom container. 

This example can be run end-to-end by executing the deploy-custom-container-mlflow.sh script in the CLI directory. 

## Define the custom container
The custom container is defined in the `mlflow.dockerfile` in this directory. The container uses the AzureML minimal inference image as its base image and creates a new conda environment from a conda yaml. For more details on adding conda environments to the AzureML minimal inference images, see the `deploy-custom-container-minimal` example. 

## Build the custom container using ACR 
Each of the two models has different conda requirements, so we build the image twice. A build argument `MLFLOW_MODEL_NAME` enables us to choose which conda yaml we want to use for our newly-created `userenv` environment.

Images can be built with ACR using the `az acr build` command. 

## Deploy to an endpoint 
We create a separate deployment for each model behind the same endpoint using the two containers. Each model has a deployment yaml that registers a separate model and environment.

## Testing the model
Traffic can be routed to a specific deployment using the header `azureml-model-deployment` or the `--deployment-name` flag of the `az ml online-endpoint invoke` CLI command. 