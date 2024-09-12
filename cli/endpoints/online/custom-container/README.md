# Azure Custom Container Examples 

This directory contains examples on how to use custom containers to deploy endpoints to Azure. In each example, a Dockerfile defines an image that may be either an extension of an Azure-originated image such as the AzureML Minimal Inference image or a third party BYOC such as Triton.


## Example Directory

Each example consists of a script located in the [CLI](../../..) directory as well as an example subdirectory that contains assets and a README.

|Example|Script|Description| 
|-------|------|---------|
|[minimal/multimodel](minimal/multimodel)|[deploy-custom-container-minimal-multimodel](../../../deploy-custom-container-minimal-multimodel.sh)|Deploy multiple models to a single deployment by extending the AzureML Inference Minimal image.|
|[minimal/single-model](minimal/single-model)|[deploy-custom-container-minimal-single-model](../../../deploy-custom-container-minimal-single-model.sh)|Deploy a single model by extending the AzureML Inference Minimal image.|
|[mlflow/multideployment-scikit](mlflow/multideployment-scikit)|[deploy-custom-container-mlflow-multideployment-scikit](../../../deploy-custom-container-mlflow-multideployment-scikit.sh)|Deploy two MLFlow models with different Python requirements to two separate deployments behind a single endpoint using the AzureML Inference Minimal Image.|
|[r/multimodel-plumber](r/multimodel-plumber)|[deploy-custom-container-r-multimodel-plumber](../../../deploy-custom-container-r-multimodel-plumber.sh)|Deploy three regression models to one endpoint using the Plumber R package|
|[tfserving/half-plus-two](tfserving/half-plus-two)|[deploy-custom-container-tfserving-half-plus-two](../../../deploy-custom-container-tfserving-half-plus-two.sh)|Deploy a simple Half Plus Two model using a TFServing custom container using the standard model registration process.|
|[tfserving/half-plus-two-integrated](tfserving/half-plus-two-integrated)|[deploy-custom-container-tfserving-half-plus-two-integrated](../../../deploy-custom-container-tfserving-half-plus-two-integrated.sh)|Deploy a simple Half Plus Two model using a TFServing custom container with the model integrated into the image.|
|[torchserve/densenet](torchserve/densenet)|[deploy-custom-container-torchserve-densenet](../../../deploy-custom-container-torchserve-densenet.sh)|Deploy a single model using a Torchserve custom container.| 
|[triton/single-model](triton/single-model)|[deploy-custom-container-triton-single-model](../../../deploy-custom-container-triton-single-model.sh)|Deploy a Triton model using a custom container|