import requests
import json
import subprocess

# 1. Add your Azure OpenAI account details
subscription = "<AOAI-ACCOUNT-SUBSCRIPTION-ID>"
resource_group = "<AOAI-ACCOUNT-RESOURCE-GROUP>"
resource_name = "<AOAI-RESOURCE-NAME>"
model_deployment_name = "<NEW-AOAI-DEPLOYMENT-NAME>"

# 2. Add the AzureML registered model name, registered model version, and the AzureML (AML) workspace path for your fine-tuned model.
# Your registered models data can be found in the `Models` tab of your AzureML workspace.
registered_model_name = "<AML-REGISTERED-MODEL-NAME>"
registered_model_version = "<AML-REGISTERED-MODEL-VERSION>"
workspace_path = "<AML-WORKSPACE-PATH-MODEL-SOURCE>"

# Run `az login` to login into your azure account in your system shell
# 3. Get Azure account access token
token = json.loads(
    subprocess.run(
        "az account get-access-token", capture_output=True, shell=True
    ).stdout
)["accessToken"]
deploy_params = {"api-version": "2023-05-01"}
deploy_headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

# 4. Set model deployment configuration. Here capacity refers to support for `1K Tokens Per Minute (TPM)` for your deployment.
deploy_data = {
    "sku": {"name": "Standard", "capacity": 1},
    "properties": {
        "model": {
            "format": "OpenAI",
            "name": f"{registered_model_name}",
            "version": f"{registered_model_version}",
            "source": f"{workspace_path}",
        }
    },
}

deploy_data = json.dumps(deploy_data)

# 5. Send PUT request to Azure cognitive services to create model deployment
request_url = f"https://management.azure.com/subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}"

r = requests.put(
    request_url, params=deploy_params, headers=deploy_headers, data=deploy_data
)

print(r)

# 6. View Your model deployment status in Azure OpenAI Studio portal
# Visit [https://oai.azure.com/portal/](https://oai.azure.com/portal/) to view your model deployments.

# 7. After your deployment succeeds, use Azure OpenAI chat playground to test your Deployment
