# Deploy 2.5-Mistral-7B model from Hugging Face using the Hugging Face TGI (Text Generation Inference) framework:

This example demonstrates how to deploy the 2.5-Mistral-7B model from Hugging Face to a managed online endpoint using the Hugging Face TGI Inference framework. This follows a similar approach to the Llama-3-8B model deployment using vLLM Inference Framework.

## How to deploy
This example can be run end-to-end using the deploy-customcontainer-tgi-huggingface.sh script located in the CLI folder.

Before running, ensure that you have sufficient GPU capacity. Additionally, add your HUGGING_FACE_HUB_TOKEN to the tgi-deployment.yml file.


## Image
The image used for this example is defined in the tgi.dockerfile. It uses huggingface/tgi-inference:latest as a base image and overrides the default ENTRYPOINT.

## Model
The model is downloaded directly from Hugging Face.

## Environment
The environment is defined inline in the deployment YAML file and references the ACR (Azure Container Registry) URL of the image. The ACR must be associated with the workspace (or have a user-assigned managed identity that enables ACRPull) in order to deploy successfully.

We define an additional environment variable called HUGGING_FACE_HUB_TOKEN, which the image uses upon initialization.

The environment also includes an inference_config block that defines the liveness, readiness, and scoring routes by path and port. Since the images used in this example are based on the AzureML Inference Minimal images, these values align with those in a non-BYOC deployment, but they must be included since we are using a custom image.