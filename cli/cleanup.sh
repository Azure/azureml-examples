ENDPOINT_LIST=$(az ml online-endpoint list --query "[*].[name]" -o tsv)
echo $ENDPOINT_LIST
for val in $ENDPOINT_LIST; do
	echo deleting $val
    `az ml online-endpoint delete -n "$val" --yes --no-wait`
done
ENDPOINT_LIST=$(az ml batch-endpoint list --query "[*].[name]" -o tsv)
echo $ENDPOINT_LIST
for val in $ENDPOINT_LIST; do
	echo deleting $val
    `az ml batch-endpoint delete -n "$val" --yes --no-wait`
done
STORAGE_ACCOUNT_LIST=$(az storage account list --query "[*].[name]" -o tsv)
echo $STORAGE_ACCOUNT_LIST
for val in $STORAGE_ACCOUNT_LIST; do
    if [[ $val == *"oepstorage"* ]]; then
        echo deleting $val
        `az storage account delete -n "$val" --yes`
    fi
done
