# setup datastore name

datastore="workspaceblobstore"



# query datastore

account=$(az ml datastore show -n $datastore --query account_name -o tsv)

account=$(echo $account|tr -d '\r')

container=$(az ml datastore show -n $datastore --query container_name -o tsv)

container=$(echo $container|tr -d '\r')



# replace storage account and container names in the YAML files

sed -i 's/account-name/'"$account"'/g' ../cli/assets/data/cloud-folder-https.yml

sed -i 's/container-name/'"$container"'/g' ../cli/assets/data/cloud-folder-https.yml

sed -i 's/account-name/'"$account"'/g' ../cli/assets/data/cloud-file-https.yml

sed -i 's/container-name/'"$container"'/g' ../cli/assets/data/cloud-file-https.yml

sed -i 's/account-name/'"$account"'/g' ../cli/assets/data/cloud-folder-wasbs.yml

sed -i 's/container-name/'"$container"'/g' ../cli/assets/data/cloud-folder-wasbs.yml

sed -i 's/account-name/'"$account"'/g' ../cli/assets/data/cloud-file-wasbs.yml

sed -i 's/container-name/'"$container"'/g' ../cli/assets/data/cloud-file-wasbs.yml

