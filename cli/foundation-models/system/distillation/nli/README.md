# Creating a Distillation Job with CLI (NLI)

### Run the Distillation CLI command pointing to the .YAML file in this folder plus the Azure ML IDs needed:

az ml job create --file ./distillation_nli.yaml --workspace-name [YOUR_AZURE_WORKSPACE] --resource-group [YOUR_AZURE_RESOURCE_GROUP] --subscription [YOUR_AZURE_SUBSCRIPTION]