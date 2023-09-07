This is a dummy pipeline job with anonymous reference for flow as component. 

Note that you will need to install a private version of cli and make sure that the compute cluster has the permission to access connections before you try to use this feature.

## dependency

To use this feature, you'll need to install a private version of `mldesigner` and cli:

```bash
python -m pip install mldesigner==0.0.103828913 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.139-py3-none-any.whl
```

## for sign

signature depends on the snapshot content, so we should use `mldesigner compile` build current flow into a sharable copy, resolving additional includes and variant in current flow:

```bash
# built flow will be under ./temp/web_classification
mldesigner compile --source ./web_classification/flow.dag.yaml --output ./temp 
```
