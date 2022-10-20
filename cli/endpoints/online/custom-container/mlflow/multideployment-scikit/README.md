# Deploy a custom container using MLFlow
This example shows how to deploy two MLFlow models that each have different conda environments using a custom container. In this example, we deploy the MLFlow models to two separate deployments (as opposed to one single deployment) to accommodate their different requirements needs. For a non-MLFlow single-deployment multimodel example, see the `deploy-custom-containerminimal-multimodel` example. 

This example can be run end-to-end by executing the deploy-custom-container-mlflow.sh script in the CLI directory. 

## Build the images
The custom container is defined in the `mlflow.dockerfile` in this directory. The container uses the AzureML minimal inference image as its base image and creates a new conda environment from a conda yaml. For more details on adding conda environments to the AzureML minimal inference images, see the `deploy-custom-container-minimal` example. 

Each of the two models has different conda requirements, so we build the image twice. A build argument `MLFLOW_MODEL_NAME` enables us to choose which conda yaml we want to use for our newly-created `userenv` environment.

Images can be built with ACR using the `az acr build` command. 

## Environment

The environment is defined inline in the deployment yaml and references the ACR url of the image. The ACR must be associated with the workspace (or have a user-assigned managed identity that enables ACRPull) in order to successfully deploy.

The environment also contains an `inference_config` block that defines the `liveness`, `readiness`, and `scoring` routes by path and port. Because the images used in this examples are based on the AzureML Inference Minimal images, these values are the same as those in a non-BYOC deployment, however they must be included since we are now using a custom image. 


## Deployment 
We create a separate deployment for each model behind the same endpoint using the two containers. Each model has a deployment yaml that registers a separate model and environment.

## Testing the model
Traffic can be routed to a specific deployment using the header `azureml-model-deployment` or the `--deployment-name` flag of the `az ml online-endpoint invoke` CLI command. 