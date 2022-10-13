# Deploy a minimal custom container
This example shows how to deploy a simple custom container endpoint by extending the AzureML Inference Server minimal cpu image. 

This example can be run end-to-end by executing the deploy-minimal-inference-custom-container.sh script in the CLI directory. 

### Build the image
Minimal.dockerfile in this directory shows how to add a new conda environment to the minimal inferencence base image. The inference image uses a conda environment called `amlenv`, however in this example we create a new conda environment called `userenv`, reinstall the `azureml-inference-server-http` package and then modify three environment variables `AZUREML_CONDA_ENVIRONMENT`, `PATH` and `LD_LIBRARY_PATH` to make our conda environment take precedence. 

Dependencies can also be added to the image through pip using a `requirements.txt` file without the need to modify environment variables. 

### Run the image locally
Containers using the AzureML inference images can be run locally by mounting the code and model directories to their expected locations. The inference images find the directories via the environment variables `AML_APP_ROOT` for the code directory and `AZUREML_MODEL_DIR` for the model directory. The default locations for thse are /var/azureml-app and /var/azureml-app/azureml-model.

### Add the image to ACR
The locally built image can either be added to ACR through pushing the locally-built image or delegating the build to ACR. 

