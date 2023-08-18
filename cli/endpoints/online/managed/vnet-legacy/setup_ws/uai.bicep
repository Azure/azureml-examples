// This is required only for the automated testign i nthe azureml-examples repo. NOT required for users fowlloing the docs.

// Parameters
@minLength(2)
@maxLength(10)
@description('Prefix for all resource names.')
param suffix string

@description('Azure region used for the deployment of all resources.')
param location string = resourceGroup().location

// Variables
var name = toLower('${suffix}')

module uai 'modules/uai.bicep' = { 
  name: 'uai-${name}-deployment'
  params: {
    location: location
    managedIdentityName: 'uai${name}'
    //id for contributor role. Reference: https://docs.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#all
    roleDefinitionIds: [
      'b24988ac-6180-42a0-ab88-20f7382dd24c'
    ]
  }
}
