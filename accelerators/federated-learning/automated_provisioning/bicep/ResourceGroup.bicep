targetScope = 'subscription'

param rgname string
param rglocation string

resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: rgname
  location: rglocation
}
