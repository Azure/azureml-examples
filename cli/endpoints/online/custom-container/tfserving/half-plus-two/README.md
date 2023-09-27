# Deploy the Half Plus Two model using TFServing
In this example, we deploy a single model (half-plus-two) using a TFServing custom container. 

This example can be run end-to-end by executing the `deploy-custom-container-tfserving-half-plus-two.sh` script in the `CLI` directory. 

## Model 
This example uses the `half-plus-two` model, which is downloaded in the script. In the deployment yaml it is registered as a model and mounted at runtime at the `AZUREML_MODEL_DIR` environment variable as in standard deployments. The default location for model mounting is `/var/azureml-app/azureml-models/<MODEL_NAME>/<MODEL_VERSION>` unless overridden by the `model_mount_path` field in the deployment yaml. 

This path is passed to TFServing as an environment variable in the deployment YAML. 

## Build the image 
This example uses the `tensorflow/serving` image with no modifications as defined in the `tfserving.dockerfile`. Although this example demonstrates the usual workflow of building the image on an ACR instance, this deployment could bypass the ACR build step and include the `docker.io` path of the image as the image URL in the deployment YAML. 

## Environment
The environment is defined inline in the deployment yaml and references the ACR url of the image. The ACR must be associated with the workspace (or have a user-assigned managed identity that enables ACRPull) in order to successfully deploy.

The environment also contains an `inference_config` block that defines the `liveness`, `readiness`, and `scoring` routes by path and port. Because the images used in this examples are based on the AzureML Inference Minimal images, these values are the same as those in a non-BYOC deployment, however they must be included since we are now using a custom image. 