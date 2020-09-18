# (PREVIEW) Deploy To Triton Inference Server with Azure Machine Learning.

This tutorial will show you how to deploy a Triton model with the Azure Machine Learning CLI. For SDK-related instructions check out the [sample notebook](aks-gpu-deploy.ipynb).

Please note this Triton Private Preview release is subject to the [Supplemental Terms of Use for Microsoft Azure Preview](https://azure.microsoft.com/support/legal/preview-supplemental-terms/).

## Prerequisites

Before attempting this tutorial, you must do this following:
* [Create an Azure Machine Learning workspace](https://docs.microsoft.com/azure/machine-learning/how-to-manage-workspace)
* [Install the Azure Machine Learning cross-platform command line interface (CLI)](https://docs.microsoft.com/azure/machine-learning/reference-azure-machine-learning-cli)

## Deploy your model with Triton

To deploy a model with Triton, run the following lines of code. These lines will create a GPU-enabled Azure Kubernetes Service (AKS) cluster and deploy a model to that cluster with Triton enabled.

The setup script assumes you have 6 cores of NCsV3 machines available in the South Central US region.
To change your VM size or location preference pass the --compute_loc or --vm_size parameters to the setup script.

```bash
python setup.py --env_name=My-Triton
az ml model register -n densenet_onnx -p ../../models/triton
az ml model deploy -n triton-densenet-onnx -m densenet_onnx:1 \
--ic inference-config.json -e My-Triton --dc deploymentconfig.json \
--overwrite --compute-target=aks-gpu

```

## Test your model

To run a sample query against your deployed model, run `python test_service.py --endpoint_name=triton-densenet-onnx`. This line will feed your deployed model (an image classification model) an image of a car and return the predicted output.

## Delete resources

Don't keep your GPU VMs running unless you are ready to use them! Run `python delete_resources.py` to delete the resources used in this tutorial.
