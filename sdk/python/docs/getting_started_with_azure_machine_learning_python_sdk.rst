.. _getting_started_with_azure_machine_learning_python_sdk:

Getting Started with Azure Machine Learning Python SDK
======================================================

This documentation provides examples and guidance on how to use the Azure Machine Learning Python SDK.

Prerequisites
-------------

1. An Azure subscription. If you don't have an Azure subscription, `create a free account <https://aka.ms/AMLFree>`_ before you begin.
2. A terminal and Python >=3.6, `<3.9 <https://pypi.org/project/azureml-core>`_.

Set up
------

Clone the Azure Machine Learning examples repository and install required packages:

.. code-block:: sh

    git clone https://github.com/Azure/azureml-examples --depth 1
    cd azureml-examples/python-sdk
    pip install --upgrade -r requirements.txt

To create or setup a workspace with the assets used in these examples, run the `setup script <setup-workspace.py>`_.

If you do not have an Azure ML workspace, run `python setup-workspace.py --subscription-id $ID`, where `$ID` is your Azure subscription id. A resource group, Azure ML workspace, and other necessary resources will be created in the subscription.

If you have an Azure ML Workspace, `install the Azure ML CLI <https://docs.microsoft.com/azure/machine-learning/reference-azure-machine-learning-cli>`_ and run `az ml folder attach -w $WS -g $RG`, where `$WS` and `$RG` are the workspace and resource group names.

Run `python setup-workspace.py -h` to see other arguments.

Getting started
---------------

To get started, see the `introductory tutorial <tutorials/an-introduction>`_ which uses Azure ML to:

- run a `"hello world"` job on cloud compute, demonstrating the basics
- run a series of PyTorch training jobs on cloud compute, demonstrating mlflow tracking & using cloud data

These concepts are sufficient to understand all examples in this repository, which are listed below.

Examples
--------

You can find the examples in the `Azure Machine Learning examples repository <https://github.com/Azure/azureml-examples>`_.

.. note:: The examples provided in the repository are for illustrative purposes and may not represent best practices for structuring Python applications.