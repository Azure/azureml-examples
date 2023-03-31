# Check if delete flag is passed
param (
    [String]$delete_resources,
    [String]$delete_files
)

#
# Define required parameters
# Update these parameters to test deployments in your own workspace
#
${subscription_id}="" # Replace with your subscription ID
${resource_group}="" # Replace with your resource group name
${workspace_name}="" # Replace with your workspace name
${registry_name}="" # Replace with your registry name
${endpoint_name}="" # Replace with your endpoint name
${deployment_name}="" # Replace with your deployment name
${model_name}="" # Name of the model to be deployed
${sku_name}="Standard_DS2_v2" # Name of the sku(instance type) Check the model-list(can be found in the parent folder(inference)) to get the most optimal sku for your model (Default: Standard_DS2_v2)


# Set default values for subscription ID, resource group, and workspace name
az account set --subscription ${subscription_id}
az configure --defaults workspace=${workspace_name} group=${resource_group}

Write-Output "---Validating Model Name ${model_name} in Registry ${registry_name}---"

# Preliminary: validate Model details and Get the latest version of the model from the registry
az ml model list --name ${model_name} --registry-name ${registry_name} > model_versions.json
If($Null -eq (Get-Content model_versions.json) -or "[]" -eq (Get-Content model_versions.json)) {
    Write-Output "---ERROR: Model name is invalid. Check the spelling of the model name---"; exit 1;
}

$versions = Get-Content .\model_versions.json -Raw | ConvertFrom-Json 
${version} = $versions[0].version


Write-Output "---Grabbing registry model $model_name version $version---"
az ml model show --name ${model_name} --version ${version} --registry-name ${registry_name} > model.json
If($Null -eq (Get-Content model.json)) {
    throw "---Model file is invalid. Check your usage of the registry command directly on powershell---"
}
$myjson = Get-Content .\model.json -Raw | ConvertFrom-Json
${model_id}=$myJson.id # Get the model ID


# Check if the endpoint already exists in the workspace
$endpoint = & {az ml online-endpoint show --name ${endpoint_name} | ConvertFrom-Json} 2>&1
if (!"$endpoint".Contains("ERROR")) {
    Write-Output "---Endpoint already exists---"
}
else {
# If it doesn't exist, create the endpoint
# Create the endpoint.yml file
    Set-Content -Path endpoint.yml -Value "`$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json`
name: ${endpoint_name}`
auth_mode: key"

    # Trigger the endpoint creation
    Write-Output "---Creating endpoint---"
    $new_endpoint = & {az ml online-endpoint create --name ${endpoint_name} --file endpoint.yml | ConvertFrom-Json} 2>&1
    if ("$new_endpoint".Contains("ERROR")) {
        Write-Output "---Endpoint creation failed---"
        Write-Output $isEndpointCreationFailed ; exit 1;
    }
    Write-Output "--Endpoint created successfully---"
}


# Create the deployment.yml file
Set-Content -Path deployment.yml -Value "`$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json`
name: default`
endpoint_name: ${endpoint_name}`
model: ${model_id}`
instance_type: ${sku_name}`
instance_count: 1"

# Trigger the deployment creation
Write-Output "---Creating deployment---"
$new_deployment = & {az ml online-deployment create --name ${deployment_name} --file deployment.yml| ConvertFrom-Json} 2>&1
if ("$new_deployment".contains("ERROR")) {
    Write-Output "---Deployment creation failed---"
    Write-Output $new_deployment; exit 1;
}
else {
    Write-Output "---Deployment created successfully---"
}


# Testing the deployment's inference if sample-request file exists
if (Test-Path -path sample-request.json) {
    Write-Output "---Inference testing---"
    Write-Output "Input: "
    Write-Output (Get-Content sample-request.json)
    $output = & {az ml online-endpoint invoke --name ${endpoint_name} --deployment-name ${deployment_name} --request-file sample-request.json > output.json} 2>&1
    if($Null -eq (Get-Content output.json)) {
        Write-Output "---Inference testing failed---"; 
        Write-Output $output; exit 1;
    }
    else {
        Write-Output "Output: "
        Write-Output (Get-Content output.json)
    }
}
else {
    Write-Output "---No sample request file found---"
}


# Delete the endpoint/deployment created if --delete_resources flag was passed
if ($delete_resources.ToLower() -eq "true") {
    if ($new_endpoint) {
        Write-Output "---Deleting endpoint/deployment---"
        $del = & {az ml online-endpoint delete --name ${endpoint_name} --yes} 2>&1
        if ("$del".Contains("ERROR")) {
            Write-Output "---Endpoint/Deployment deletion failed---"; 
            Write-Output $del; exit 1;
        }
        Write-Output "---Endpoint/Deployment deleted successfully---"
    }
    else {
        Write-Output "---Deleting deployment---"
        $del = & {az ml online-deployment delete --name ${deployment_name} --endpoint-name ${endpoint_name} --yes} 2>&1
        if ("$del".Contains("ERROR")) {
            Write-Output "---Deployment deletion failed---"; 
            Write-Output $del; exit 1;
        }
        Write-Output "---Deployment deleted successfully---"
    }
}


# Delete the files downloaded/created if the --delete_files flag was passed
if ($delete_files.ToLower() -eq "true") {
    Write-Output "---Deleting files---"
    $files_to_delete = ".\AzCopy", ".\AzCopy.zip", ".\mlflow_model_folder", ".\model_versions.json", ".\model.json", ".\endpoint.yml", ".\deployment.yml", ".\output.json"
    Foreach ($file in $files_to_delete) {
        If (Test-Path $file) {
            Remove-Item $file -Recurse  
        }
    }
    Write-Output "---Files deleted successfully---"
}
