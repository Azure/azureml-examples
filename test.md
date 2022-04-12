subscription_id="e9b2ec51-5c94-4fa8-809a-dc1e695e4896"
resource_group='clwantest'
workspace="clwan-centraluseuap"


subscription_id="74eccef0-4b8d-4f83-b5f9-fa100d155b22"
resource_group='AmlComponentNotebook'
workspace="Aml-Component-Notebook-EUS"


az account set -s 74eccef0-4b8d-4f83-b5f9-fa100d155b22
az configure --defaults group="AmlComponentNotebook" workspace="Aml-Component-Notebook-EUS"



az account set -s 96aede12-2f73-41cb-b983-6d11a904839b
az configure --defaults group="cli-examples" workspace="master"



conda activate cliv2
$Env:AZURE_EXTENSION_DIR="D:\enlistment\aml\sdk\sdk-cli-v2\src\cli\src";$Env:AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=$true



subscription_id = 'd128f140-94e6-4175-87a7-954b9d27db16'
resource_group = 'clwantest'
workspace = 'hod-canary'



subscription_id = '96aede12-2f73-41cb-b983-6d11a904839b'
resource_group = 'sdk'
workspace = 'sdk-canary'


subscription_id = '96aede12-2f73-41cb-b983-6d11a904839b'
resource_group = 'sdk'
workspace = 'sdk-master'

az account set -s 96aede12-2f73-41cb-b983-6d11a904839b
az configure --defaults group="sdk" workspace="sdk-westus2"