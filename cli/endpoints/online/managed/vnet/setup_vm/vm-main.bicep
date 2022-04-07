// Used to create  scoring VM for automated internal  tests. Not intented a user script.

@description('Azure region used for the deployment of all resources.')
param location string = resourceGroup().location

@description('name of the test scoring vm')
param vmName string = 'moevnet-vm'

@description('Name of the user assigned identity (UAI). UAI is used to login through az cli from within the vm to create image in ACR, deploy model in azure ml etc')
param identityName string = 'test-vm-uai'

@description('name of the vnet to deploy the vm')
param vnetName string

@description('name of the subnet to deploy the vm')
param subnetName string = 'snet-scoring'

resource uaiResourceId 'Microsoft.ManagedIdentity/userAssignedIdentities@2018-11-30' existing = {
  name: identityName
}

resource vnet 'Microsoft.Network/virtualNetworks@2021-05-01' existing = {
  name: vnetName  
}

resource subnetResourceId 'Microsoft.Network/virtualNetworks/subnets@2021-05-01' existing = {
  name: subnetName
  parent: vnet
}


@description('Set of tags to apply to all resources.')
param tags object = {}

module vm 'modules/vm.bicep' = {
  name: '${vmName}-deployment'  
  params:{
    location: location
    tags: tags
    vmName: vmName
    virtualMachineSize: 'Standard_F2s_v2'
    uaiResourceId: uaiResourceId.id
    subnetId: subnetResourceId.id
  }
}
