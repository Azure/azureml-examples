# Deploy multiple models to one deployment

This example shows how to deploy multiple models to a single deployment by extending the AzureML Inference Minimal CPU image.

It can be run end-to-end by executing the `deploy-custom-container-minimal-multimodel.sh` script in the CLI directory. 

## Build the image

The image used for this example is defined in the `minimal-multimodel.dockerfile` and uses the pip method of adding additional requirements needed to score the model. To add requirements to an AzureML Inference base image using conda, see the `deploy-custom-container-minimal-single-model` example, which demonstrates both methods. 

## Models

The two models used in this example are included in the `models` subdirectory. They are both sklearn models and need the same requirements. The two models are registered as a single "registered model" on Azure by including them in the same directory and passing that directory as the `path` of the model. In this example, the model is defined inline in the deployment. 

## Score script

The score script for this example is located in the `code` directory. 

At deployment time, both models are loaded into memory using the standard method of using the `AZUREML_MODEL_DIR` environment variable. Even though this is a custom container deployment, the `AZUREML_MODEL_DIR` is still present in the container. This happens any time a model is mounted to an endpoint container, even if the custom container does not use an Azure image as a base image.

The models are loaded into a dictionary keyed on their names. When a request is received, the desired model is retrieved from the JSON payload, and the relevant model is used to score the payload. 

## Environment

The environment is defined in the deployment yaml and references the ACR url of the image. The ACR must be associated with the workspace (or have a user-assigned managed identity that enables ACRPull) in order to successfully deploy. Both models share the same environment. 

The environment also contains an `inference_config` block that defines the `liveness`, `readiness`, and `scoring` routes by path and port. Because the images used in this examples are based on the AzureML Inference Minimal images, these values are the same as those in a non-BYOC deployment, however they must be included since we are now using a custom image. 

### Python requirements

Both models are loaded in the same process by the same Python interpreter. Models loaded in a multi-model deployment using this technique must share the same Python version and - even if they do not have the same dependencies in a strict sense (i.e. a scikit model loaded alongside a torch model) - their dependiences must not conflict with each other and be able to be simultaneously imported. 
