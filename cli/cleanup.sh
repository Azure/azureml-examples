ENDPOINT_LIST=$(az ml endpoint list --query "[*].[name]" -o tsv)
echo $ENDPOINT_LIST
for val in $ENDPOINT_LIST; do
	echo deleting $val
    `az ml endpoint delete -n "$val" --yes --no-wait`
done
