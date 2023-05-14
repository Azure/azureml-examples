[![smoke](https://github.com/Azure/azureml-examples/workflows/smoke/badge.svg)](https://github.com/Azure/azureml-examples/actions/workflows/smoke.yml)
[![Python code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
# **Training a GPT model with NeMo Megatron**

This example will focus on pretraining a GPT model using the NeMo-Megatron framework using the PILE dataset on AzureML.

## **Setup**
### **Hardware**
A100 GPUs are recommended for this job.

#### **Linear Scaling with Infiniband Enabled SKUs**
To attain linear scaling for large model, one important step can be to use InfiniBand. InfiniBand enables low-latency, GPU-to-GPU communication across nodes in a cluster. InfiniBand requires specialized hardware to operate. Only some VM SKUs on Azure contain this required hardware. You can view the full list of InfiniBand-enabled machine SKUs [here](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-hpc#rdma-capable-instances). 

### **Environment**
The environment found at ``src/environment`` creates an instance of the Nvidia NeMo container and includes all of the needed packages to run NeMo jobs on AzureML. Note that this is not the latest version of the container. To create the environment, run the following command in an AzureML compute instance terminal:
```
az ml environment create --file ./src/environment/env.yml
```
## **Code**
All of the code described in this document can be found either in the files in this repository or in the NeMo Repository. Example is originally sourced from the [NeMo Repository](https://github.com/NVIDIA/NeMo).

## **Preprocess Data**
### **Download and Register Data**
Before preprocessing the data, the dataset needs to be downloaded and stored as a data asset to be used in the preprocessing job. The PILE dataset can be downloaded [here](https://the-eye.eu/public/AI/pile/). This example trains using just the first two partitions, but more can be used if desired.

For this example, wget and zstd were used in the terminal to download and extract the data, but any method can be used.
```
pip install wget
pip install zstd
wget https://the-eye.eu/public/AI/pile/train/00.jsonl.zst
wget https://the-eye.eu/public/AI/pile/train/01.jsonl.zst
zstd -d 00.jsonl.zst
zstd -d 01.jsonl.zst
```

After downloading and extracting the data, Two files should be available: ``00.jsonl`` and ``01.jsonl``. Place these files into a folder and name the folder ``PILE-data``. This folder can then be registered using the ``register-data.yaml`` file included with this example and running the following command in your terminal:
```
az ml data create -f register-data.yaml
```
>NOTE: This will take some time.

These raw json files can now be used for the data preprocessing job.

### **Preprocessing**
Before running the preprocessing job, some parameters will need to be adjusted. In the ``NeMo-preprocess-data.yaml`` file, change the following parameters to suit your configuration.
- ``num-workers``Set this to the total number of gpus being used to preprocess the data.
- ``compute`` Set this to the name of your compute.
- ``process_count_per_instance`` Set this to the number of GPUs in each node.
- ``instance_count`` Set this to the number of nodes.

Run the following command to begin data preprocessing:
```
az ml job create --file NeMo-preprocess-data.yaml
```

Once completed, the preprocessed data will be located in the ``outputs`` folder of the job.

## **Run the Job**
Before running the job, some parameters will need to be adjusted. In the ``NeMo-train-5B.yaml`` file, change the following parameters to suit your configuration.
- ``trainer.devices`` Set this to the number of GPUs that exist per node.
- ``trainer.num_nodes`` Set this to the number of nodes available for the job.
- ``path: azureml://datastores/workspaceartifactstore/paths/ExperimentRun/dcid.brave_ghost_bxh654n2dd/outputs`` Replace this path with the path to the preprocessed data. After preprocessing the data in the above step, the ``outputs`` folder from above should be located in the artifactstore of your workspace. The example path shown can be used but replace ``dcid.brave_ghost_bxh654n2dd`` with ``dcid.<id of the data preprocessing job>``.
- ``compute`` Set this to the name of your compute.
- ``process_count_per_instance`` Set this to the number of GPUs in each node.
- ``instance_count`` Set this to the number of nodes.

Run the following command to start the job:
```
az ml job create --file NeMo-train-5B.yaml
```
## **Results**

