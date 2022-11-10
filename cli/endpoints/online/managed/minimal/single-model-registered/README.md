# Deploy an already-registered model to a Managed Online Example 
In this example, a model is registered before deploying an endpoint rather than defining the local model inline. The model is then included in the deployment YAML as a reference to the registered model. 

To run this example, please execute the [script](../../../../../deploy-moe-minimal-single-model-registered.sh) in the CLI folder. 

# Model
The model is registered using the `az ml model create -f` command and passing the `model.yml` file. This example uses a model with a local path, thereby uploading the model assets to Azure at registration time. However, a path to a datastore may also be used to register a model that already exists in Azure Storage. 

# Deployment
In an inline-model deployment, the `model` block is multiple lines and may contain sub-fields such as name, version, and path. Since the model is already registered ahead of deployment creation, the model block is replaced with a one-line reference to the registered model in the format `azureml:<MODEL_NAME>:<MODEL_VERSION>`. 
