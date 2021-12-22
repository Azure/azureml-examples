Param(
    [Parameter(Mandatory=$true,
    HelpMessage="The guid of the subscription where the orchestrator will live.")]
    [string]
    $SubscriptionId_Orchestrator,
    [Parameter(Mandatory=$true,
    HelpMessage="The name of the orchestrator AML workspace.")]
    [string]
    $AMLWorkspaceName,
    [Parameter(Mandatory=$true,
    HelpMessage="The name of the resource group containing orchestrator AML workspace.")]
    [string]
    $AMLWorkspaceRGName,
    [Parameter(Mandatory=$false,
    HelpMessage="The location of the orchestrator AML workspace (if it does not exist already).")]
    [string]
    $AMLWorkspaceLocation="westus2",
    [Parameter(Mandatory=$false,
    HelpMessage="The name of the K8s cluster to connect.")]
    [string]
    $K8sClusterName="cont-k8s-01",
    [Parameter(Mandatory=$false,
    HelpMessage="The name of the Azure ML compute to be created.")]
    [string]
    $AMLComputeName="cont-01-compute"    
)

# load useful functions
. "$PSScriptRoot\AzureUtilities.ps1"

# Validating the required name of the Azure ML compute
Confirm-ComputeName $AMLComputeName

# making sure we're in the right subscription
Write-Output "We'll be setting up the orchestrator in this subscription: $SubscriptionId_Orchestrator."
az account set --subscription $SubscriptionId_Orchestrator
# log in
az login

# create the orchestrator workspace if it does not exist already
$Workspaces =  az ml workspace list --resource-group $AMLWorkspaceRGName --query "[?name=='$AMLWorkspaceName']" | ConvertFrom-Json
if ($Workspaces.Length -eq 0){
    Write-Output "Name of the AML workspace's resource group to create: $AMLWorkspaceRGName, in $AMLWorkspaceLocation location."
    Deploy-RGIfInexistent $AMLWorkspaceRGName, $AMLWorkspaceLocation, "AML workspace"
    Write-Output "Creating the workspace '$AMLWorkspaceName'..."
    az deployment group create --resource-group $AMLWorkspaceRGName --template-file .\bicep\AMLWorkspace.bicep --parameters workspacename=$AMLWorkspaceName  location=$AMLWorkspaceLocation
} else {
    Write-Output "The AML workspace $AMLWorkspaceName already exists."
}

# Connect K8s cluster to Azure Arc
# 1. Register providers for Azure Arc-enabled Kubernetes
Write-Output "Registering the required providers for Azure Arc-enabled K8s..."
az provider register --namespace Microsoft.Kubernetes
az provider register --namespace Microsoft.KubernetesConfiguration
az provider register --namespace Microsoft.ExtendedLocation
# (waiting for successful registration)
Wait-SuccessfulRegistration "Microsoft.Kubernetes"
Wait-SuccessfulRegistration "Microsoft.KubernetesConfiguration"
Wait-SuccessfulRegistration "Microsoft.ExtendedLocation"
# 2. Create a resource group if it doesn't exist already
$ArcClusterName = $K8sClusterName + "-arc"
$ArcClusterRGName = $ArcClusterName + "-rg"
$ArcClusterLocation = $AMLWorkspaceLocation # we will create the Arc RG in the same location as the AML workspace RG (arbitrary)
Deploy-RGIfInexistent $ArcClusterRGName $ArcClusterLocation "Arc cluster"
# 3. Connect to the existing K8s cluster
Write-Output "Connecting to the existing K8s cluster..." # the existing K8s cluster is determined by the contents of the kubeconfig file, which is created at the end of the CreateK8sCluster.ps1 script
az connectedk8s connect --name $ArcClusterName --resource-group $ArcClusterRGName

# Deploy the Azure ML extension to the Arc cluster
Write-Output "Deploying the Azure ML extension to the Arc cluster..."
az k8s-extension create --name arcml-extension --extension-type Microsoft.AzureML.Kubernetes --config enableTraining=True --cluster-type connectedClusters --cluster-name $ArcClusterName --resource-group $ArcClusterRGName --scope cluster --auto-upgrade-minor-version False

# Attach the Arc cluster to the Azure ML workspace
Write-Output "Attaching the Arc cluster to the Azure ML workspace..."
$ResourceId = "/subscriptions/" + $SubscriptionId_Orchestrator +"/resourceGroups/" + $ArcClusterRGName + "/providers/Microsoft.Kubernetes/connectedClusters/" + $ArcClusterName
az ml compute attach --type kubernetes --name $AMLComputeName --workspace-name $AMLWorkspaceName --resource-group $AMLWorkspaceRGName --resource-id $ResourceId
