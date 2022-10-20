# Deploy Huggingface models using Torchserve

This example demonstrates how to deploy Huggingface models to a managed online endpoint and follows along with the [Serving Huggingface Transformers using TorchServe](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers) example from HuggingFace. 

In this example we deploy a BERT model for text generation. 

## How to deploy

This example can be run end-to-end using the `deploy-customcontainer-torchserve-huggingface-textgen.sh` script in the `CLI` folder. Torchserve is not required to be installed. 

## Image

The image used for this example is defined in file `ts-hf-tg.dockerfile`. It uses `pytorch/torchserve` as a base image and overrides the default `CMD` so that the `model-store` points to the location of the mounted model upon initialization by referencing the `AZUREML_MODEL_DIR` env var and that the `models` loaded are defined in the custom env var `TORCHSERVE_MODELS`. 

## Model

To prepare the model, the [Huggingface_Transformers](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers) directory is cloned from the `pytorch/serve` Github repo. We use the same image built for deployment above to prepare the model per the instructions in the Huggingface example. 

## Environment
The environment is defined inline in the deployment yaml and references the ACR url of the image. The ACR must be associated with the workspace (or have a user-assigned managed identity that enables ACRPull) in order to successfully deploy.

We define an additional env var called `TORCHSERVE_MODELS` which is used by the image upon initialization. 

The environment also contains an `inference_config` block that defines the `liveness`, `readiness`, and `scoring` routes by path and port. Because the images used in this examples are based on the AzureML Inference Minimal images, these values are the same as those in a non-BYOC deployment, however they must be included since we are now using a custom image. 