# setup datastore name
datastore="workspaceblobstore"

# query datastore
account=$(az ml datastore show -n $datastore --query account_name -o tsv)
account=$(echo $account|tr -d '\r')
container=$(az ml datastore show -n $datastore --query container_name -o tsv)
container=$(echo $container|tr -d '\r')

# replace storage account and container names in the YAML files
sed -i 's/account-name/'"$account"'/g' ./create_from_uri/cloud-folder-https-Dataset.yml
sed -i 's/container-name/'"$container"'/g' ./create_from_uri/cloud-folder-https-Dataset.yml
sed -i 's/account-name/'"$account"'/g' ./create_from_uri/cloud-file-https-Dataset.yml
sed -i 's/container-name/'"$container"'/g' ./create_from_uri/cloud-file-https-Dataset.yml
sed -i 's/account-name/'"$account"'/g' ./create_from_uri/cloud-folder-wasbs-Dataset.yml
sed -i 's/container-name/'"$container"'/g' ./create_from_uri/cloud-folder-wasbs-Dataset.yml
sed -i 's/account-name/'"$account"'/g' ./create_from_uri/cloud-file-wasbs-Dataset.yml
sed -i 's/container-name/'"$container"'/g' ./create_from_uri/cloud-file-wasbs-Dataset.yml
