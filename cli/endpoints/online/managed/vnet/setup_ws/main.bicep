// Execute this main file to configure Azure Machine Learning end-to-end in a moderately secure set up

// Parameters
@minLength(2)
@maxLength(10)
@description('Prefix for all resource names.')
param suffix string

@description('Azure region used for the deployment of all resources.')
param location string = resourceGroup().location

@description('[Optional] Azure region used for appinsights.')
param appinsightsLocation string = location

@description('Set of tags to apply to all resources.')
param tags object = {}

@description('Virtual network address prefix')
param vnetAddressPrefix string = '192.168.0.0/16'

@description('private endpoint subnet address prefix')
param scoringSubnetPrefix string = '192.168.0.0/24'

// Variables
var name = toLower('${suffix}')

// Virtual network and network security group
module nsg 'modules/nsg.bicep' = { 
  name: 'nsg-${name}-deployment'
  params: {
    location: location
    tags: tags 
    nsgName: 'nsg-${name}'
  }
}

module vnet 'modules/vnet.bicep' = { 
  name: 'vnet-${name}-deployment'
  params: {
    location: location
    virtualNetworkName: 'vnet-${name}'
    networkSecurityGroupId: nsg.outputs.networkSecurityGroup
    vnetAddressPrefix: vnetAddressPrefix
    scoringSubnetPrefix: scoringSubnetPrefix
    tags: tags
  }
}

// Dependent resources for the Azure Machine Learning workspace
module keyvault 'modules/keyvault.bicep' = {
  name: 'kv-${name}-deployment'
  params: {
    location: location
    keyvaultName: 'kv${name}'
    keyvaultPleName: 'ple-${name}-kv'
    subnetId: '${vnet.outputs.id}/subnets/snet-scoring'
    virtualNetworkId: '${vnet.outputs.id}'
    tags: tags
  }
}

module storage 'modules/storage.bicep' = {
  name: 'st${name}-deployment'
  params: {
    location: location
    storageName: 'st${name}'
    storagePleBlobName: 'ple-${name}-st-blob'
    storagePleFileName: 'ple-${name}-st-file'
    storageSkuName: 'Standard_LRS'
    subnetId: '${vnet.outputs.id}/subnets/snet-scoring'
    virtualNetworkId: '${vnet.outputs.id}'
    tags: tags
  }
}

module containerRegistry 'modules/containerregistry.bicep' = {
  name: 'cr${name}-deployment'
  params: {
    location: location
    containerRegistryName: 'cr${name}'
    containerRegistryPleName: 'ple-${name}-cr'
    subnetId: '${vnet.outputs.id}/subnets/snet-scoring'
    virtualNetworkId: '${vnet.outputs.id}'
    tags: tags
  }
}

module applicationInsights 'modules/applicationinsights.bicep' = {
  name: 'appi-${name}-deployment'
  params: {
    location: appinsightsLocation
    applicationInsightsName: 'appi-${name}'
    tags: tags
  }
}

module azuremlWorkspace 'modules/machinelearning.bicep' = {
  name: 'mlw-${name}-deployment'
  params: {
    // workspace organization
    machineLearningName: 'mlw-${name}'
    machineLearningFriendlyName: 'Private link endpoint sample workspace'
    machineLearningDescription: 'This is an example workspace having a private link endpoint.'
    location: location
    tags: tags

    // dependent resources
    applicationInsightsId: applicationInsights.outputs.applicationInsightsId
    containerRegistryId: containerRegistry.outputs.containerRegistryId
    keyVaultId: keyvault.outputs.keyvaultId
    storageAccountId: storage.outputs.storageId

    // networking
    subnetId: '${vnet.outputs.id}/subnets/snet-scoring'
    virtualNetworkId: '${vnet.outputs.id}'
    machineLearningPleName: 'ple-${name}-mlw'

  }
  dependsOn: [
    keyvault
    containerRegistry
    applicationInsights
    storage
  ]
}

