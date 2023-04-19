---
page_type: sample
languages:
- azurecli
- python
products:
- azure-machine-learning
description: An official sample showcasing use of on-behalf-of feature in AzureML. Using this feature customers can use their AAD identity from within training script to perform any operations only limited by their access, like creating another AzureML Job or reading secrets from a key vault in a secure way.
---

# AzureML - On Behalf of Feature
AzureML On Behalf of (OBO) is a powerful feature which allows AzureML users to use their AAD Identity within the training script of a remote Job (a job that runs on a remote compute). 

## Why should you use it ?

AzureML makes your AAD identity available inside training script. Any resource you can access when running code on your machine can be accessed from the training script running on a remote compute.

## How do I use AzureML On Behalf of (OBO) ?

There are 2 things that are required to use OBO feature:
- Specify in Job definition you want to use AzureML OBO.
- Use `AzureMLOnBehalfOfCredential` credential class in training script

### Step 1: Specify in Job definition I want to use AzureML OBO
This is as easy as adding below section to your job definition:

```yaml
identity:
  type: user_identity
```

[Job.yaml](job.yaml) from [on behalf of](../on-behalf-of/) shows how a Job definition specifies to use OBO feature.

### Step 2: Use `AzureMLOnBehalfOfCredential` credential class in training script

`AzureMLOnBehalfOfCredential` credential class in a part of `azure-ai-ml` package and can be used with any client that accepts credential class from `azure-identity` package. In your training script use this credential class with client of resources you would like to access.

Code snipped below shows how `AzureMLOnBehalfOfCredential` can be used to access azure key vault.
```python
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.keyvault.secrets import SecretClient


credential = AzureMLOnBehalfOfCredential()
secret_client = SecretClient(vault_url="https://my-key-vault.vault.azure.net/", credential=credential)
secret = secret_client.get_secret("secret-name")
```

[Training script](../on-behalf-of/src/aml_run.py) from [on behalf of](../on-behalf-of/) sample show how `AzureMLOnBehalfOfCredential` is used to create a job in AzureML from within the training script.
