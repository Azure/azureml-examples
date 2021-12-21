@description('Specifies the name of the workspace.')
param workspacename string
@description('Specifies the location of the Azure Machine Learning workspace and dependent resources.')
param location string = resourceGroup().location
@description('Specifies whether to reduce telemetry collection and enable additional encryption.')
param hbi_workspace bool = false

var tenantId = subscription().tenantId
var storageAccountName_var = replace('st-${workspacename}','-','') // replace because only alphanumeric characters are supported
var keyVaultName_var = 'kv-${workspacename}'
var applicationInsightsName_var = 'appi-${workspacename}'
var containerRegistryName_var = replace('cr-${workspacename}','-','') // replace because only alphanumeric characters are supported
var workspaceName_var = workspacename
var storageAccount = storageAccountName.id
var keyVault = keyVaultName.id
var applicationInsights = applicationInsightsName.id
var containerRegistry = containerRegistryName.id

resource storageAccountName 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: storageAccountName_var
  location: location
  sku: {
    name: 'Standard_RAGRS'
  }
  kind: 'StorageV2'
  properties: {
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    supportsHttpsTrafficOnly: true
  }
}
resource keyVaultName 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: keyVaultName_var
  location: location
  properties: {
    tenantId: tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: []
    enableSoftDelete: true
  }
}
resource applicationInsightsName 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName_var
  location: (((location == 'eastus2') || (location == 'westcentralus')) ? 'southcentralus' : location)
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}
resource containerRegistryName 'Microsoft.ContainerRegistry/registries@2021-08-01-preview' = {
  sku: {
    name: 'Standard'
  }
  name: containerRegistryName_var
  location: location
  properties: {
    adminUserEnabled: true
  }
}
resource workspaceName 'Microsoft.MachineLearningServices/workspaces@2021-07-01' = {
  identity: {
    type: 'SystemAssigned'
  }
  name: workspaceName_var
  location: location
  properties: {
    friendlyName: workspaceName_var
    storageAccount: storageAccount
    keyVault: keyVault
    applicationInsights: applicationInsights
    containerRegistry: containerRegistry
    hbiWorkspace: hbi_workspace
  }
  dependsOn: [
    storageAccountName
    keyVaultName
    applicationInsightsName
    containerRegistryName
  ]
}
