# new types bug bash

## Prep steps
1. Uninstall existing extension and reinstall the latest cli packages for PuP
    **refer to [CLI channel](https://teams.microsoft.com/l/channel/19%3a0e07499759514cedb184f47791c95014%40thread.tacv2/CLI?groupId=13e1702e-ccb0-4b25-aff8-d637a92c658b&tenantId=72f988bf-86f1-41af-91ab-2d7cd011db47) for latest package version**
    ```cmd
        az extension list
        az extension remove -n azureml-cli-ml
        az extension remove -n ml
        az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.58822429-py3-none-any.whl --yes
    ```
2. Configure the bugbash worksace as default workspace
    ```cmd
    az account set -s "60582a10-b9fd-49f1-a546-c4194134bba8"
    az configure --defaults group="datasettest-canary" workspace="qianie-v2-bugbash-master"
    ```

3. Job submission cmd
    ```cmd
    C:\Users\qianie\Source\Repos\azureml-examples\cli\jobs\runtime-bugbash\lightgbm\iris> az ml job create --file .\job.yml
    ```
    
4. Testing scenarios
    1) uri_file, uri_folder, mltable, mlflow_model as input happy path in command job
        - different modes(mount, download, direct)
        - different uri formats for input
    2) uri_file, uri_folder, mltable, mlflow_model as output happy path in command job
        - output only supports short form datastore uri path

    2) edge cases
        - path doesnt existing
        - special chars in path
        - multile uri_file to the same folder

## Links
```cmd
Quick Read for March Breaking Changes

Branch for CLI changes – march-cli-preview
Branch for SDK changes – march-sdk-preview
Branch for Doc Updates - release-preview-aml-cli-v2-refresh
Updated YAML samples for reference - online - Repos (visualstudio.com)
Bug Template for issues - link
How to install new SDK and CLI?
# download SDK and CLI wheels from here

Link to private wheels


#remove any existing CLI version

az extension remove -n ml

#add new version of CLI

az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.58822429-py3-none-any.whl --yes


#remove old version of SDK

pip uninstall azure-ml


#add new version of SDK

pip install azure-ml==0.0.58822429 --extra-index-url  https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
```
