param location string
param containerName string
param uaiId string
param subnetId string

var numberCpuCores = 2
var memory = 4
var imageName = 'mcr.microsoft.com/azure-cli:latest'

var ports = [
  {
      'port': 80
      'protocol': 'TCP'
  }
]


resource containerName_resource 'Microsoft.ContainerInstance/containerGroups@2021-09-01' = {
  location: location
  name: containerName
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${uaiId}': {}
    }
  }
  properties: {
    containers: [
      {
        name: containerName        
        properties: {
          image: imageName
          command: [
            '/bin/bash'
            '-c'
            'az extension add -n ml -y' 
            'az login --identity -u ${uaiId} --allow-no-subscriptions; sleep infinity'
          ]
          resources: {
            requests: {
              cpu: numberCpuCores
              memoryInGB: memory
            }
          }
          ports: ports
        }
      }
    ]
    restartPolicy: 'OnFailure'
    osType: 'Linux'
    ipAddress: {
      type: 'Private'
      ports: ports
    }
    subnetIds: [
      {
        id: subnetId
      }
    ]
  }
  tags: {}  
}

