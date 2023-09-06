This is a dummy pipeline job with anonymous reference for flow as component. 

Note that you will need to install a private version of cli and make sure that the compute cluster has the permission to access connections before you try to use this feature.

+ for sign

signature depends on the snapshot content, so we should use `mldesigner compile` build current flow into a sharable copy, resolving additional includes and variant in current flow:

To use this feature, you'll need to install a private version of `mldesigner`:

```bash
python -m pip install mldesigner==0.0.103828913 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2

# built flow will be under ./temp/web_classification
mldesigner compile --source ./web_classification/flow.dag.yaml --output ./temp 
```
