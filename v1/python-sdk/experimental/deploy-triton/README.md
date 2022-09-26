---
page_type: sample
languages:
- python
- azurecli
products:
- azure-machine-learning
description: Learn how to efficiently deploy to GPUs with the [Triton inference server](https://github.com/triton-inference-server/server) and Azure ML.
experimental: in preview
---

# Real-time inference on GPUs in Azure Machine Learning

**Note**: this tutorial is experimental and prone to failure

The notebooks in this directory show how to take advantage of the interoperability between Azure Machine Learning and [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) for cost-effective real time inference on GPUs.

## Python instructions

Open either of the sample notebooks in this directory to run Triton in Python.

## CLI instructions

You must have the latest version of the Azure Machine Learning CLI installed to run these commands.
Follow the [instructions here](https://docs.microsoft.com/azure/machine-learning/reference-azure-machine-learning-cli#prerequisites) to download or upgrade the CLI.

```{bash}
python src/model_utils.py
az ml model register -p models/triton -n bidaf-model --model-framework=Multi
az ml model deploy -n triton-webservice -m bidaf-model:1 --dc deploymentconfig.json --compute-target aks-gpu-deploy
```

Once you have deployed, try querying the model metadata endpoint:

```{bash}
# Get the scoring URI
az ml service show --name triton-webservice
# Get the keys
az ml service get-keys --name triton-webservice
curl -H "Authorization: Bearer <primaryKey>" -v <scoring-uri>v2/ready
```

Read more about the [KFServing predict API here](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md).
