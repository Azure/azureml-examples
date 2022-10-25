# Deploy a Torchserve custom container using the Densenet model

In this example, we deploy a single model (Densenet) using a Torchserve custom container. 

This example can be run end-to-end by executing the `deploy-custom-container-torchserve-densenet.sh` script in the `CLI` directory. 

## Model

This example uses the `densenet161` model. It is registered as a model ahead-of-time using a separate call to `az ml model create`. Upon instantiation, the `AZUREML_MODEL_DIR` environment variable as in standard deployments. The default location for model mounting is `/var/azureml-app/azureml-models/<MODEL_NAME>/<MODEL_VERSION>` unless overridden by the `model_mount_path` field in the deployment yaml. 

This path is passed to TFServing as an environment variable in the deployment YAML. 

## Environment
The environment is defined inline in the deployment yaml and references the ACR url of the image. The ACR must be associated with the workspace (or have a user-assigned managed identity that enables ACRPull) in order to successfully deploy.

The environment also contains an `inference_config` block that defines the `liveness`, `readiness`, and `scoring` routes by path and port. Because the images used in this examples are based on the AzureML Inference Minimal images, these values are the same as those in a non-BYOC deployment, however they must be included since we are now using a custom image. 