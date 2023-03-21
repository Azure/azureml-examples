## Optimized Environment for large scale distributed training

To effectively run optimized and significantly faster training and inference for large models on AzureML, we recommend the new Azure Container for PyTorch (ACPT) environment which includes the best of Microsoft technologies for training with PyTorch on Azure. In addition to AzureML packages, this environment includes latest Training Optimization Technologies: [Onnx / Onnx Runtime / Onnx Runtime Training](https://onnxruntime.ai/),
[ORT MoE](https://github.com/pytorch/ort/tree/main/ort_moe), [DeepSpeed](https://www.deepspeed.ai/), [MSCCL](https://github.com/microsoft/msccl), Nebula checkpointing and others to significantly boost the performance. 

## Azure Container for PyTorch (ACPT)

### Curated Environment

There are multiple ready to use curated images published with latest pytorch, cuda versions, ubuntu for [ACPT curated environment](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments#azure-container-for-pytorch-acpt-preview). You can find the ACPT curated environments by filtering by “ACPT” in the Studio:

![image](https://user-images.githubusercontent.com/39776179/217119432-0418209c-d8e9-49c6-b47d-3612a517e47b.png)

Once you’ve selected a curated environment that has the packages you need, you can refer to it in your YAML file. For example, if you want to use one of the ACPT curated environments, your command job YAML file might look like the following:

```sh
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
description: Trains a 175B GPT model
experiment_name: "large_model_experiment"
compute: azureml:cluster-gpu
code: ../../src
environment: azureml:AzureML-ACPT-pytorch-1.12-py39-cuda11.6-gpu@latest
environment_variables:
  NCCL_DEBUG: 'WARN'
  NCCL_DEBUG_SUBSYS: 'WARN'
  CUDA_DEVICE_ORDER: 'PCI_BUS_ID'
  NCCL_SOCKET_IFNAME: 'eth0'
  NCCL_IB_PCI_RELAXED_ORDERING: '1'
```

You can also use SDK to specify the environment name
```sh
job = command(
    
    environment="AzureML-ACPT-pytorch-1.12-py39-cuda11.6-gpu@latest"
)
```
 @latest tag to the end of the environment name will pull the latest image. If you want to be specific about the curated environment version number, you can specify it using the following syntax:
```sh
 $schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
...
environment: azureml:AzureML-ACPT-pytorch-1.12-py39-cuda11.6-gpu:3
...
```

### Custom Environment
If you are looking to extend curated environment and add HF transformers or datasets to be installed, you can create a new env with docker context containing ACPT curated environment as base image and additional packages on top of it as below:

![image](https://user-images.githubusercontent.com/39776179/217162558-235fe518-734d-4b89-8940-71dd4744dda1.png)

In new docker context, use curated env of ACPT as base image and add pip install of transformers, datasets and others.
![image](https://user-images.githubusercontent.com/39776179/217162413-643ef5ce-ebee-4dfe-bc42-c6b7fa60250b.png)

In addition, you can also save dockerfile to your local path in a environment.yaml
```sh
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: custom_aml_environment
build:
  path: docker
```
Create the custom environment using:
```sh
az ml environment create -f cloud/environment/environment.yml
```

```sh
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
description: Trains a 175B GPT model
experiment_name: "large_model_experiment"
compute: azureml:cluster-gpu

code: ../../src
environment: azureml:custom_aml_environment@latest
```
You can find more detail at [Custom Environment using SDK and Cli](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?tabs=cli#create-an-environment)

## Benefits

Benefits of using the ACPT curated environment include: 

- Significantly faster training and inference when using ACPT environment.
- Optimized Training framework to set up, develop, accelerate PyTorch model on large workloads. 
- Up-to-date stack with the latest compatible versions of Ubuntu, Python, PyTorch, CUDA\RocM, etc.   
- Ease of use: All components installed and validated against dozens of Microsoft workloads to reduce setup costs and accelerate time to value  
- Latest Training Optimization Technologies: Onnx / Onnx Runtime / Onnx Runtime Training, ORT MoE, DeepSpeed,  MSCCL, and others.. 
- Integration with Azure ML: Track your PyTorch experiments on ML Studio or using the AzureML SDK  
- As-IS use with pre-installed packages or build on top of the curated environment  
- The image is also available as a DSVM 
- Azure Customer Support 
