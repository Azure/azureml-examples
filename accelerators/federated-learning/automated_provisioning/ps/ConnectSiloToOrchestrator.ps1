Param(
    [Parameter(Mandatory=$true,
    HelpMessage="The guid of the subscription where we'll be preparing the silo.")]
    [string]
    $SubscriptionId,
    [Parameter(Mandatory=$true,
    HelpMessage="The name of the K8s cluster to create (this will NOT be the name of the attached compute in the Azure ML portal).")]
    [string]
    $K8sClusterName,
    [Parameter(Mandatory=$false,
    HelpMessage="The region where to create the K8s cluster (westus2 by default).")]
    [string]
    $RGLocation="westus2",
    [Parameter(Mandatory=$false,
    HelpMessage="The number of agents in the K8s cluster (1 by default).")]
    [int]
    $AgentCount=1,
    [Parameter(Mandatory=$false,
    HelpMessage="The agent VM SKU ('Standard_B4ms' by default).")]
    [string]
    $AgentVMSize="Standard_B4ms"
)

# # making sure we're in the right subscription
# Write-Output "We'll be setting up a silo in this subscription: $SubscriptionId."
# az account set --subscription $SubscriptionId
# # log in
# az login

#az group create --name thopo-arc-test --location westus2 --output table
#az connectedk8s connect --name thopo-arc-test-1 --resource-group thopo-arc-test

# az k8s-extension create --name arcml-extension --extension-type Microsoft.AzureML.Kubernetes --config enableTraining=True --cluster-type connectedClusters --cluster-name thopo-arc-test-1 --resource-group thopo-arc-test --scope cluster --auto-upgrade-minor-version False


#az ml compute attach --type kubernetes --name thopo-k8s-arm --workspace-name aml1p-ml-wus2 --resource-group aml1p-rg --resource-id /subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/thopo-arc-test/providers/Microsoft.Kubernetes/connectedClusters/thopo-arc-test-1