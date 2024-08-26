# Deploy Phi-3-mini-4k-instruct model from Huggingface using vLLM Inferencing Framework

This example demonstrates how to deploy Llama-3-8B model from Huggingface to a managed online endpoint and follows along with the [Llama-3-8B model from Huggingface using vLLM Inferencing Framework](https://github.com/pytorch/serve/tree/master/examples/vllm). 

## How to deploy
This example can be run end-to-end using the `deploy-customcontainer-vllm-huggingface-test.sh` script in the `CLI` folder. 

Before running please insure that you have GPU capacity. Also, please add your HUGGING_FACE_HUB_TOKEN to the vllm-deployment.yml. 


## Image
The image used for this example is defined in file `vllm.dockerfile`. It uses `vllm/vllm-openai:latest` as a base image and overrides the default `ENTRYPOINT`. 

## Model
The model is downloaded from huggingface. 

## Environment
The environment is defined inline in the deployment yaml and references the ACR url of the image. The ACR must be associated with the workspace (or have a user-assigned managed identity that enables ACRPull) in order to successfully deploy.

We define an additional env var called `THUGGING_FACE_HUB_TOKEN` which is used by the image upon initialization. 

The environment also contains an `inference_config` block that defines the `liveness`, `readiness`, and `scoring` routes by path and port. Because the images used in this examples are based on the AzureML Inference Minimal images, these values are the same as those in a non-BYOC deployment, however they must be included since we are now using a custom image. 