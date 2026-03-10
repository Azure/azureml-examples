# Introduction

Hyperparameter settings could have a big impact on the model performance, but typically, the process is quite tedious and resource-consuming, which involves training multiple models using same algorithm and training data, but with different hyperparameters, and then evaluate the model from each training run to determine the best-performing model with respect to a performance metric.

So, we need to perform large number of experiments with different parameters, and we know that Azure ML allows us to accumulate all experiment results, including performance metrics, in one place: the Azure ML Workspace. So, basically all we need to do is to submit a bunch of experiments with different hyperparameters and let HyperDrive do the heavy lifting for us in an automatic way.

There are other markdown documents and notebooks about using HyperDrive to tune hyperparameters in AzureML, including [Hyperparameter tuning a model with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters), [Azure Machine Learning Pipeline with HyperDriveStep](https://github.com/Azure/MachineLearningNotebooks/blob/467630f95583a88b731ebc4bdefff1cc525df054/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-parameter-tuning-with-hyperdrive.ipynb), and [Train, hyperparameter tune, and deploy with PyTorch](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/pytorch/train-hyperparameter-tune-deploy-with-pytorch/train-hyperparameter-tune-deploy-with-pytorch.ipynb). However, their approach is useful when the user originally designs the pipeline based on HyperDrive step. In many applications, on the other hand, there is a pipeline already in place and the user looks for adding the HyperDrive step as an extension to their current pipeline with minimum changes and modifications.

This asset is a solution to such requests from AzureML users. The pipeline used in this asset is very general with three common steps of training, evaluation, and registration, which are found in almost every AzureML pipeline. We explain how to use HyperDrive step in connection with these three main steps. 

# Fictitious Model

One of the challenges in the practice of hyperparameter tuning is the computational resources it consumes, where for even a simple classification model, the running time and computational cost could be a concern. To ease this difficulty, and to focus on the HyperDrive step and its specific components only, we have created a [fictitious model](./Fictitious_Model_for_Hyperparameter_Tuning.ipynb) based on a set of deterministic calculations that emulate the behavior of a machine learning algorithm with changing its key parameters. While mimicking some key characteristics of an actual machine learning model, the model runs extremely fast even on a standard CPU-based compute cluster. Neither it requires a dataset and the steps related to it, as we believe there already exist plenty of resources that explain those parts very well.

The proposed model generates four outputs that mimic the following performance metrics for a hypothetical classifier:

* Precision
* Recall
* Training loss
* Model accuracy

These four outputs are affected by three input variables that mimic the following tuning parameters for the classifier: 

* Threshold
* Number of epochs
* Learning rate  

# Getting Started

The only file that you need to run is the [HyperDrive_MultiStep_Training_Pipeline](./HyperDrive_MultiStep_Training_Pipeline.ipynb) notebook in this folder. To run the notebook, the only requirements are to provide:

* A `config.json` file to access the AzureML workspace artifacts that you can [download from your AzureML](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace) portal, which includes the following items:
  * subscription_id
  * resource_group
  * workspace_name
* A `.env` file that will be utilized by the python dotenv library to read in secrets through the notebook, with the following information required:
  * AML_CLUSTER_CPU_SKU: AzureML compute cluster used to distribute the training across a cluster of CPU or GPU compute nodes in the cloud. See the list of [VM SKUs]((https://docs.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list) ) supported for AzureML managed online endpoints, and learn how to create and manage a [compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target#azure-machine-learning-compute-managed) in your AzureML workspace.
  * AML_CLUSTER_NAME: The name of AzureML compute cluster used for running a job.
  * AML_CLUSTER_PRIORITY: You may choose to use [low-priority VMs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-optimize-cost) with reduced price compared to dedicated VMs.
  * AML_CLUSTER_MIN_NODES: Control the minimum nodes available to the cluster. To avoid charges when no jobs are running, set the minimum nodes to 0, which allows Azure ML to de-allocate the nodes when they aren't in use. Any value larger than 0 will keep that number of nodes running, even if they are not in use.
  * AML_CLUSTER_MAX_NODES: The max number of nodes to auto-scale up to when running a job on the cluster.
  * AML_CLUSTER_SCALE_DOWN: Node idle time in seconds before scaling down the cluster, defaults to 1800.
  * BLOB_DATASTORE_NAME: The name of [Azure Blob storage](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction) for storing data.
  * STORAGE_NAME: The name of the datastore for running a job.
  * STORAGE_KEY: Access keys of the storage account for running a job, defaults to None.
  * STORAGE_CONTAINER: The name of the Azure Blob container for running a job.

To make the notebook more concise and readable, helper scripts are used within the following directory structure:

```bash
├───using-hyperdrive
    │   HyperDrive_MultiStep_Training_Pipeline.ipynb
    │   Fictitious_Model_for_Hyperparameter_Tuning.ipynb
    │   README.md
    │   .env        
    └───src
        ├───common
        │      attach_compute.py
        │      get_datastores.py
        │      model_helpers.py  
        ├───model
        │      algorithms.py
        └───pipeline 
               evaluate.py
               register.py
               train.py           
```

By adoption, this general structure could also help create more complex models and pipelines. 

# Build and Run

The [run object](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py) in the notebook provides the interface to the run history while the job is running and after it has completed. As the run is executed, it goes through the following stages:

* **Preparing:** A docker image is created according to the environment defined. The image is uploaded to the workspace's container registry and cached for later runs. Logs are also streamed to the run history and can be viewed to monitor progress. 
* **Scaling:** The cluster attempts to scale up if the Batch AI cluster requires more nodes to execute the run than are currently available.
* **Running:** All scripts in the script folder are uploaded to the compute target, data stores are mounted or copied, and the script is executed. Outputs from stdout and the `./logs` folder are streamed to the run history and can be used to monitor the run.
* **Post-Processing:** The `./outputs` folder of the run is copied over to the run history.

Once you've trained the model, you can register it to your workspace. Model registration lets you store and version your models in your workspace to simplify model management and deployment.
