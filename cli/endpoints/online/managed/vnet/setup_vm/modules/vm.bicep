param location string

@description('VM resource name')
param vmName string

@description('Tags to add to the resources')
param tags object = {}

// random user name and sshk public key. The VM is not private IP enabled and accessed only using "az vm run-command invoke"
param adminUserName string = 'd3F4gfddfE4'
var randomSshPublicKey = 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCUCu8YZlOF4bPPu4I8lXBB1Yv9xtqQZlLzGcvgbLRUeU2/GJh2EIWS331bsQX72Ww8NZgqnvvqLz5yRak8Y171zmokkpUrNQUGLEUT5BATF8s3ssNnAksNRDPqZoweUjk4JFKQ4TmvgWMW8QbEHpc4x3urfNxHHet3xKLm5I4pqd35/UUB/5moE1YjjUQLsurcjJCrUEmn3+X2itO7TS8DM4q3FhnNasAOq3UFILBNWYQisrrPbj62yt1BNFmz2yq7uBxYpNiDzeKZvGVT17de5yKNh1v9F99QG62hApjIAH+GDW4G5tm92W0Q4sf6LdC7fE3/BvNmYjVNt6JCpEJx'

var networkInterfaceName = 'nic${vmName}'
param subnetId string

param virtualMachineSize string
param uaiResourceId string
param ubuntuSku string = '20_04-lts-gen2'


resource networkInterfaceName_resource 'Microsoft.Network/networkInterfaces@2021-03-01' = {
  name: networkInterfaceName
  location: location
  tags: tags
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: {
            id: subnetId
          }
          privateIPAllocationMethod: 'Dynamic'
        }
      }
    ]
  }
  dependsOn: []
}

resource vmName_resource 'Microsoft.Compute/virtualMachines@2021-07-01' = {
  name: vmName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${uaiResourceId}': {}
    }
  }
  properties: {
    hardwareProfile: {
      vmSize: virtualMachineSize
    }
    storageProfile: {
      osDisk: {
        createOption: 'FromImage'
        managedDisk: {
          storageAccountType: 'Premium_LRS'
        }
        deleteOption: 'Delete'
      }
      imageReference: {
        publisher: 'canonical'
        offer: '0001-com-ubuntu-server-focal'
        sku: ubuntuSku
        version: 'latest'
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: networkInterfaceName_resource.id
          properties: {
            deleteOption: 'Delete'
          }
        }
      ]
    }
    osProfile: {
      computerName: 'cname${vmName}'
      adminUsername: adminUserName
      linuxConfiguration: {
        patchSettings: {
          patchMode: 'ImageDefault'         
        }
        disablePasswordAuthentication: true        
        ssh: {
          publicKeys: [
            {
              keyData: randomSshPublicKey
              path: '/home/${adminUserName}/.ssh/authorized_keys'
            }
          ]
        }
      }
    }
  }
}

