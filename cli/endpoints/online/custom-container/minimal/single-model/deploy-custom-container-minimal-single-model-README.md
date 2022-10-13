# Deploy a minimal custom container
This example shows how to deploy a simple custom container endpoint by extending the AzureML Inference Server minimal cpu image in two different ways - via a conda file or via pip requirements. 

It can be run end-to-end by executing the `deploy-minimal-inference-custom-container.sh` script in the CLI directory. 

## Build the image(s)
The subdirectories `conda-in-dockerfile` and `pip-in-dockerfile` correspond to two different techniques for adding custom Python requirements at build time to the AzureML Inference base image. Requirements can also be added at deployment time via pip or conda through environment variables. Alternatively, environments can be defined on Azure as a combination of an inference base image and a conda file. In this case, Azure will manage the actual build. 

### Using pip
Adding pip requirements to a Dockerfile can be accomplished through a simple `pip install` statement as in `minimal-single-model-pip-in-dockerfile.dockerfile` or via a `requirements.txt` file copied in during the build process. No additional environment variables are needed. 

### Conda

The file `minimal-single-model-conda-in-dockerfile.dockerfile` in the `conda-in-dockerfile` directory shows how to add a new conda environment to the minimal inferencence base image. The inference image uses a conda environment called `amlenv`, however in this example we create a new conda environment called `userenv`, reinstall the `azureml-inference-server-http` package and then modify three environment variables `AZUREML_CONDA_ENVIRONMENT`, `PATH` and `LD_LIBRARY_PATH` to make our conda environment take precedence. The conda file used for this image is located at `endpoints/online/model-1/environment`. 

## Model 

Both deployments uses the `model-1` sklearn model located at `endpoints/online/model-1/model` as well as the sample request located in the parent directory. It is defined inline in the deployment yaml and mounted at deployment time. 

## Score script

The scoring script for this model is located at `endpoints/online/model-1/onlinescoring` and is identitcal to the scoring script for this model that would be used in a non-custom container scenario. 

Even though this deployment uses a custom container, the `AZUREML_MODEL_DIR` environment variable is still present in the container, and so the model init process that uses it is the same as well. 

## Environment

The environment is defined inline in the deployment yaml and references the ACR url of the image. The ACR must be associated with the workspace (or have a user-assigned managed identity that enables ACRPull) in order to successfully deploy.

The environment also contains an `inference_config` block that defines the `liveness`, `readiness`, and `scoring` routes by path and port. Because the images used in this examples are based on the AzureML Inference Minimal images, these values are the same as those in a non-BYOC deployment, however they must be included since we are now using a custom image. 

