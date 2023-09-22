This is a dummy pipeline job with anonymous reference for flow as component. 

Note that you will need to install a private version of cli and make sure that the compute cluster has the permission to access connections before you try to use this feature.

## dependency

To use this feature, you'll need to install a private version of `mldesigner` and cli:

```bash
# install the private version azure-ai-ml first. mldesigner depends this to compile flow as component
python -m pip install azure-ai-ml==1.10.0a20230904003 --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/
# install the private version mldesigner
python -m pip install mldesigner[promptflow]==0.0.105430780 --extra-index-url https://azuremlsdktestpypi.azureedge.net/test-sdk-cli-v2

az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.139-py3-none-any.whl
```

## for sign

signature depends on the snapshot content, so we should use `mldesigner compile` build current flow into a sharable copy, resolving additional includes and variant in current flow:

```bash
# built flow will be under ./temp/web_classification
mldesigner compile --source ./web_classification/flow.dag.yaml --output ./temp 
```
