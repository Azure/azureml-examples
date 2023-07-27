.. _working_with_azure_machine_learning_endpoints:

Working with Azure Machine Learning Endpoints
==============================================

Azure Machine Learning provides a variety of ways to deploy your models and manage endpoints. This includes using the Azure Machine Learning CLI, SDK, and UI. 

Azure Machine Learning Deployment
---------------------------------

Azure Machine Learning helps you deploy your models with managed endpoints. This can be achieved using the Azure Machine Learning CLI. You need to provide the YAML file as shown in the figure below:

.. image:: imgs/AMLCLI.png
   :width: 300

The YAML file contains the name, environment, and target machines which can be CPU/GPU based. Azure Machine Learning recognizes Triton model format, which means if the directory structure of your model repository follows the correct syntax in terms of model and its' config file, it should be running with Triton by default.

There's also a UI-based option where you can upload the model from your local workstation and can then see if it's in Triton format. 

.. image:: imgs/AMLUI1.png
   :width: 300

.. image:: imgs/AMLUI2.png
   :width: 300

Once the model is uploaded, you can see the optimized config as well as the models and their different versions in the "artifacts" section. You can now create managed endpoints for real-time inferencing using the generated URLs.

Azure Machine Learning Python SDK Examples
------------------------------------------

The Azure Machine Learning Python SDK provides a variety of examples to help you get started with Azure Machine Learning. These examples include running a "hello world" job on cloud compute, running a series of PyTorch training jobs on cloud compute, and more. You can find these examples in the `azureml-examples` repository on GitHub.

Efficient Data Loading for Large Training Workload
-------------------------------------------------

When training AI models, it's important to ensure that the GPU on your compute is fully utilized to keep costs as low as possible. Serving training data to the GPU in a performant manner goes a long way to ensure you can fully utilize the GPU. Azure ML offers two options for storage: Azure Blob - Standard (HDD) and Azure Blob - Premium (SSD). The choice of storage can impact the performance of your training workload.

Running AutoML Jobs with CLI
----------------------------

Azure Machine Learning also supports running AutoML jobs using the CLI. This includes running jobs for time-series forecasting, text NER, text classification (multi-label and multi-class), and regression. The CLI commands for running these jobs are provided in the respective sections of the `azureml-examples` repository.

Megatron-DeepSpeed
------------------

Megatron-DeepSpeed is a complex example that adds additional support for several features such as MoE model training, Curriculum Learning, 3D Parallelism, and easy-to-use examples for training on Azure. You can find more information about running this example on AzureML in the `Megatron-DeepSpeed` repository on GitHub.

.. note:: The information provided in this documentation is based on the content provided in the `azureml-examples` repository on GitHub. For more detailed information and examples, please refer to the original source.