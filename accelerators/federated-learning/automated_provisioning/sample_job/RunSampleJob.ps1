Param(
    [Parameter(Mandatory=$false,
    HelpMessage="The guid of the subscription to which the orchestrator belongs.")]
    [string]
    $SubscriptionId="48bbc269-ce89-4f6f-9a12-c6f91fcb772d",
    [Parameter(Mandatory=$false,
    HelpMessage="The name of the orchestrator AML workspace.")]
    [string]
    $WorkspaceName="aml1p-ml-wus2",
    [Parameter(Mandatory=$false,
    HelpMessage="The name of the orchestrator AML workspace.")]
    [string]
    $ResourceGroup="aml1p-rg"  
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
