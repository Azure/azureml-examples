// Creates a machine learning workspace, private endpoints and compute resources

@description('Azure region of the deployment')
param location string

@description('Tags to add to the resources')
param tags object

@description('Machine learning workspace name')
param machineLearningName string

@description('Machine learning workspace display name')
param machineLearningFriendlyName string = machineLearningName

@description('Machine learning workspace description')
param machineLearningDescription string

@description('Resource ID of the application insights resource')
param applicationInsightsId string

@description('Resource ID of the container registry resource')
param containerRegistryId string

@description('Resource ID of the key vault resource')
param keyVaultId string

@description('Resource ID of the storage account resource')
param storageAccountId string

@description('Resource ID of the subnet resource')
param subnetId string

@description('Resource ID of the virtual network')
param virtualNetworkId string

@description('Machine learning workspace private link endpoint name')
param machineLearningPleName string

resource machineLearning 'Microsoft.MachineLearningServices/workspaces@2021-07-01' = {
  name: machineLearningName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    // workspace organization
    friendlyName: machineLearningFriendlyName
    description: machineLearningDescription

    // dependent resources
    applicationInsights: applicationInsightsId
    containerRegistry: containerRegistryId
    keyVault: keyVaultId
    storageAccount: storageAccountId

    // disable public network access
    publicNetworkAccess: 'Enabled'
  }
}

module machineLearningPrivateEndpoint 'machinelearningnetworking.bicep' = {
  name: 'machineLearningNetworking'
  scope: resourceGroup()
  params: {
    location: location
    tags: tags
    virtualNetworkId: virtualNetworkId
    workspaceArmId: machineLearning.id
    subnetId: subnetId
    machineLearningPleName: machineLearningPleName
  }
}

output machineLearningId string = machineLearning.id
