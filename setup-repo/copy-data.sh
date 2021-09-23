# setup variables
datapath="example-data"
datastore="workspaceblobstore"

# query workspace
subscription=$(az account show --query id -o tsv)
group=$(az ml workspace show --query resource_group -o tsv)

# query datastore
account=$(az ml datastore show -n $datastore --query account_name -o tsv)
container=$(az ml datastore show -n $datastore --query container_name -o tsv)
endpoint=$(az ml datastore show -n $datastore --query endpoint -o tsv)
protocol=$(az ml datastore show -n $datastore --query protocol -o tsv)

# add contributor access to datastore
az ad signed-in-user show --query objectId -o tsv | az role assignment create \
    --role "Storage Blob Data Owner" \
    --assignee @- \
    --scope "/subscriptions/$subscription/resourceGroups/$group/providers/Microsoft.Storage/storageAccounts/$account"

# copy iris data
azcopy copy "https://azuremlexamples.blob.core.windows.net/datasets/iris.csv" "$protocol://$account.blob.$endpoint/$container/$datapath/"

# copy diabetes data
azcopy copy "https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv" "$protocol://$account.blob.$endpoint/$container/$datapath/"

# copy mnist data
azcopy copy "https://azuremlexamples.blob.core.windows.net/datasets/mnist" "$protocol://$account.blob.$endpoint/$container/$datapath/" --recursive

# copy cifar data
azcopy copy "https://azuremlexamples.blob.core.windows.net/datasets/cifar-10-python.tar.gz" "$protocol://$account.blob.$endpoint/$container/$datapath/"

