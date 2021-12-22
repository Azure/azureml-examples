Param(
    [Parameter(Mandatory=$true,
    HelpMessage="The guid of the subscription to which the orchestrator belongs.")]
    [string]
    $SubscriptionId,
    [Parameter(Mandatory=$true,
    HelpMessage="The name of the orchestrator AML workspace.")]
    [string]
    $WorkspaceName,
    [Parameter(Mandatory=$true,
    HelpMessage="The name of the orchestrator AML workspace resource group.")]
    [string]
    $ResourceGroup  
)

# making sure we're in the right subscription
az account set --subscription $SubscriptionId
# log in
az login

# Create the mnist dataset from local files if it doesn't exist
$DatasetName = "mnist_test"
$ExistingMNISTDataset = az ml dataset list --resource-group $ResourceGroup --workspace-name $WorkspaceName --name $DatasetName
if ($ExistingMNISTDataset.Length -eq 0){
    Write-Output "The dataset '$DatasetName' does not exist. Creating it from the local files..."
    az ml dataset create --resource-group $ResourceGroup --workspace-name $WorkspaceName --name $DatasetName --local-path "./sample_job/src/mnist_data/"
} else {
    Write-Output "The dataset '$DatasetName' already exists."
}

# run the job
Write-Output "Submitting the job..."
az ml job create -f ./sample_job/job.yml --web
