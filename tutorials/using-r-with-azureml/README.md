# 

## Pre-requisities

* azure machine learning workspace created
* azure-cli installed in local environment
* az ml install installed in local environment
* R installed in local environment
* yaml package

## 01-setup

* 
* create compute instance
    * Compute name needs to be unique across all existing computes within an Azure region
    * https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-compute-instance
    ```yml
    $schema: https://azuremlschemas.azureedge.net/latest/computeInstance.schema.json
    name: minimal-example-i
    type: computeinstance
    ```
    * Can't use setupscipt via cli or rest api, only Studio or ARM Template 

* create compute cluster 
    * https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-compute-aml
    ```yml
    $schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json 
    name: minimal-example
    type: amlcompute
    ```