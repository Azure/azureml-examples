# find online endpoints
ONLINE_ENDPOINT_LIST=$(az ml online-endpoint list --query "[*].[name]" -o tsv)
echo $ONLINE_ENDPOINT_LIST

# find batch endpoints
BATCH_ENDPOINT_LIST=$(az ml batch-endpoint list --query "[*].[name]" -o tsv)
echo $BATCH_ENDPOINT_LIST

# find storage accounts
STORAGE_ACCOUNT_LIST=$(az storage account list --query "[*].[name]" -o tsv)
echo $STORAGE_ACCOUNT_LIST

# find compute instances
CI_LIST=$(az ml compute list --type ComputeInstance --query "[*].[name]" -o tsv)
echo $CI_LIST

# fine UAI
NAME_LIST=$(az identity list --query "[].{name:name}" -o tsv | sort -u)
echo $NAME_LIST

# find left over autoscale settings created for online endpoints
AUTOSCALE_SETTINGS_LIST=$(az monitor autoscale list  --query "[*].[name]" -o tsv)
echo $AUTOSCALE_SETTINGS_LIST

# find workspaces created via testing
WORKSPACES_LIST=$(az ml workspace list --query "[*].[name]" -o tsv)
echo $WORKSPACES_LIST

# Wait for 2 hours so that we don't delete entities that are still in use.
echo waiting
sleep 2h

# delete online endpoints
echo deleting online endpoints
for val in $ONLINE_ENDPOINT_LIST; do
    echo deleting $val
    `az ml online-endpoint delete -n "$val" --yes --no-wait`
done

# delete batch endpoints
echo deleting online endpoints
for val in $BATCH_ENDPOINT_LIST; do
    echo deleting $val
    `az ml batch-endpoint delete -n "$val" --yes --no-wait`
done

# delete storage accounts
echo deleting storage accounts
for val in $STORAGE_ACCOUNT_LIST; do
    if [[ $val == *"oepstorage"* ]]; then
        echo deleting $val
        `az storage account delete -n "$val" --yes`
    fi
done

# delete compute instances
echo deleting compute instances
for val in $CI_LIST; do
    echo deleting $val
    `az ml compute delete -n "$val" --yes --no-wait`
done

# delete UAI
echo deleting user identities
for name in $NAME_LIST; do
    if [[ $name == *"oep-user-identity"* ]]; then
        echo deleting $name
        `az identity delete --name "$name"`
    fi
done

# delete left over autoscale settings created for online endpoints
echo deleting autoscale settings
for val in $AUTOSCALE_SETTINGS_LIST; do
    if [[ $val == autoscale-* ]]; then
        echo deleting $val
        `az monitor autoscale delete -n "$val"`
    fi
done

# delete workspaces created via testing
echo deleting workspaces
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
