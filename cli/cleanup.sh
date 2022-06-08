# delete online endpoints
ENDPOINT_LIST=$(az ml online-endpoint list --query "[*].[name]" -o tsv)
echo $ENDPOINT_LIST
for val in $ENDPOINT_LIST; do
    echo deleting $val
    `az ml online-endpoint delete -n "$val" --yes --no-wait`
done

# delete batch endpoints
ENDPOINT_LIST=$(az ml batch-endpoint list --query "[*].[name]" -o tsv)
echo $ENDPOINT_LIST
for val in $ENDPOINT_LIST; do
    echo deleting $val
    `az ml batch-endpoint delete -n "$val" --yes --no-wait`
done

# delete storage accounts
STORAGE_ACCOUNT_LIST=$(az storage account list --query "[*].[name]" -o tsv)
echo $STORAGE_ACCOUNT_LIST
for val in $STORAGE_ACCOUNT_LIST; do
    if [[ $val == *"oepstorage"* ]]; then
        echo deleting $val
        `az storage account delete -n "$val" --yes`
    fi
done

# delete compute instances
CI_LIST=$(az ml compute list --type ComputeInstance --query "[*].[name]" -o tsv)
echo $CI_LIST
for val in $CI_LIST; do
    echo deleting $val
    `az ml compute delete -n "$val" --yes --no-wait`
done

# delete UAI
NAME_LIST=$(az identity list --query "[].{name:name}" -o tsv | sort -u)
echo $NAME_LIST
for name in $NAME_LIST; do
    if [[ $name == *"oep-user-identity"* ]]; then
        echo deleting $name
        `az identity delete --name "$name"`
    fi
done

# delete left over autoscale settings created for online endpoints
AUTOSCALE_SETTINGS_LIST=$(az monitor autoscale list  --query "[*].[name]" -o tsv)
for val in $AUTOSCALE_SETTINGS_LIST; do
    if [[ $val == autoscale-* ]]; then
        echo deleting $val
        `az monitor autoscale delete -n "$val"`
    fi
done

# delete workspaces created via testing
WORKSPACES_LIST=$(az ml workspace list --query "[*].[name]" -o tsv)
for val in $WORKSPACES_LIST; do
    if [[ $val == "mlw-"* ]]; then
        if [[ $val == "mlw-mevnet" ]]; then
            echo skipping $val
        else
            echo deleting $val
            `az ml workspace delete -n "$val" --yes --no-wait --all-resources`
        fi
    else
        echo $val not a match
    fi
done
