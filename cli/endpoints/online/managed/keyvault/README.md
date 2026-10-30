# Access Keyvault secrets from a Managed Online Endpoint

This example can be run end-to-end with the script ['deploy-moe-keyvault.sh'](../../../../deploy-moe-keyvault.sh).

## Overview
In this example we create a Keyvault, set a secret, and then retrieve the secret from a Managed Online Endpoint using the endpoint's system-assigned managed identity. By using the managed identity, the need to pass secrets as well as any other credentials in the image or deployment is avoided.

## Prerequisites
* Azure subscription. If you don't have an Azure subscription, sign up to try the [free or paid version of Azure Machine Learning](https://azure.microsoft.com/free/) today.

* Azure CLI and ML extension. For more information, see [Install, set up, and use the CLI (v2) (preview)](how-to-configure-cli.md).

## Create an endpoint

```bash
az ml online-endpoint create -n $ENDPOINT_NAME 
```

## Create a Keyvault
Due to the absence of the `--no-self-perms` flag, the Keyvault automatically assigns all permissions to the current user or SP.

```bash 
az keyvault create -n $KV_NAME -g $RESOURCE_GROUP
```

### Set access policy
The Principal ID of the endpoint is needed to grant it the `GET` permission on the Keyvault. 
```bash
ENDPOINT_PRINCIPAL_ID=$(az ml online-endpoint show -n $ENDPOINT_NAME --query identity.principal_id -o tsv)
az keyvault set-policy -n $KV_NAME --object-id $ENDPOINT_PRINCIPAL_ID --secret-permissions get
```

## Set a secret
The secret called `multipler` is set to `7`. 
```bash
az keyvault secret set --vault-name $KV_NAME -n multiplier --value 7
```

## Create a Deployment

The scoring script uses a `ManagedIdentityCredential` to authenticate itself to the Keyvault via a `SecretClient` from the `azure-keyvault` package. No arguments are needed to instantiate the credential object when this code is executed in a deployment, because it reads the environment variables `MSI_SECRET` and `MSI_ENDPOINT` which are already present.

As part of the deployment, we will pass an environment variable called `KV_SECRET_MULTIPLIER` and give it the value `multiplier@https://<VAULT_NAME>.vault.azure.net`. The convenience function `load_secrets` looks for environment variables with `KV_SECRET` and replaces their values with the actual value of the secret from the keyvault. 

When a request is received, `input` is multiplied by our secret. 

```yml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: kvdep
endpoint_name: <ENDPOINT_NAME>
model:
  path: "."
code_configuration:
  code: code
  scoring_script: score.py
environment:
  image: mcr.microsoft.com/azureml/minimal-py312-inference:latest
  conda_file: env.yml
environment_variables:
  KV_SECRET_MULTIPLIER: multiplier@https://<KV_NAME>.vault.azure.net
instance_type: Standard_DS3_v2
instance_count: 1
```

```bash
az ml online-deployment create \
  -f endpoints/online/managed/keyvault/keyvault-deployment.yml \
  --set endpoint_name=$ENDPOINT_NAME \
  --set environment_variables.KV_SECRET_MULTIPLIER="multiplier@https://$KV_NAME.vault.azure.net" \
  --all-traffic
```

## Test the endpoint
```bash
az ml online-endpoint invoke -n $ENDPOINT_NAME \
  --request-file endpoints/online/managed/keyvault/sample_request.json
``` 

## Delete assets

```bash
az ml online-endpoint delete --yes -n $ENDPOINT_NAME --no-wait
```

```bash
az keyvault delete --name $KV_NAME --no-wait
```