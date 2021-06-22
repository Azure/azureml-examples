---
page_type: sample
languages:
- azurecli
products:
- azure-machine-learning
description: Learn how to deploy an R model as an Azure Machine Learning managed online endpoint
---

# Deploying R models as Azure Machine Learning managed online endpoints

This folder contains the assets that are called from [deploy-r.sh](../../../../deploy-r.sh) to deploy an R model as a managed online endpoint in Azure Machine Learning. This README explains how to modify the assets in this folder to deploy your own R model.

## Overview

We deploy R models using a feature called [custom containers](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-custom-container), which lets you bring a Docker container and deploy it as a managed online endpoint. In the R case, we Dockerize your model using [plumber](https://www.rplumber.io/) and its associated [Docker image](https://www.rplumber.io/articles/hosting.html#default-dockerfile). See the included [Dockerfile](./Dockerfile) and [plumber script](./scripts/plumber.R) for more details.

## Deploying your own model

To deploy your own model, do the following:

### Place model in scripts folder

Assuming you've saved your model as a .rda or .rds file, save it in the `scripts` folder in this directory. This directory is "mounted" to your Docker container when we deploy the container as an online endpoint, so you can change the contents of this directory without needing to rebuild your Docker container.

### Modify plumber.R to load your saved model

Modify the third function in plumber.R to load the saved model and run the model every time the endpoint is invoked. If your model takes more or fewer inputs, you may need to change the function signature. For example, if you have a model that takes three inputs, update the function decorator with the line `@param c The third number to add` and also update the function signature to say `function(a, b, c)`.

### Create an endpoint in your own subscription

Follow the steps in [our documentation](https://docs.microsoft.com/azure/machine-learning/how-to-configure-cli) to configure the 2.0 CLI. Then run [deploy-r.sh](../../../../deploy-r.sh) (if running on a Linux machine). Alternatively, call `az ml endpoint create -n r-endpoint.yml`.

### Call into your deployed endpoint

You can now follow the steps [here](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-managed-online-endpoints#invoke-the-endpoint-to-score-data-with-your-model) to send data to your deployed endpoint.