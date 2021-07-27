
We are introducing a new and improved way of building Machine Learning Pipelines in AzureML. The new experience is based on Components that form the building blocks of composable and reusable Pipelines. Components enable users specify an input/output interface, execution environment, the command to run and other settings required to run a script. Users can register these Components with the AzureML Workspace and use them across various Pipelines without having to worry about the details and dependencies needed to run the scripts. Registered Components can be used in Pipelines written in YAML and submitted from CLI, authored in Python and created using the drag-n-drop Designer UI in AzureML Studio, offering a consistent Pipelines experience across CLI, SDK and UI surfaces.

### Highlights 
* <u>Better MLOps with YAML based Pipelines and Components authoring</u>: A declarative YAML based Pipeline definition enables clear separation between the orchestration and infrastructure code from the data science code that can use languages like Python or R. 
* <u>Collaborate and share ML workflow assets with Components</u>: Unlike Pipelines v1 in which you couldn't easily use a Pipeline Step written for one Pipeline in another Pipeline, Components are not bound Pipelines but are assets that can be independently checked-into Git repos, registered with Workspaces and used across different Pipelines. 
* <u>Manage Component lifecycle and trigger Pipeline Jobs with CLI 2.0</u>: YAML based Components and Pipelines is part of the broader CLI 2.0 wave of features which means that YAML syntax and terminology is consistent with other improvements and new features shipping with CLI 2.0. 

### What's available today?
* Components to run Python scripts, R scripts or any Command (aka Command Component) with YAML/CLI. 
* Pipeline Job to orchestrate Command Components. 

### What's coming next? 
* Python and Designer UI support for authoring Pipeline Jobs with Components.
* More components - Sweep, AutoML, Distributed Training, Batch, etc. 
* Subgraphs (aka Pipeline Components) for large and complex Pipelines.

### How to get started?
Pre-requisites:
1. AzureML Workspace with a compute cluster. We strongly recommend using an existing test or sandbox Workspace or creating a new Workspace because the private preview bits can have bugs. DO NOT TRY THE PREVIEW ON A WORKSPACE WITH PRODUCTION ASSETS.
2. If you do not have the Azure CLI installed, follow the installation instructions at https://docs.microsoft.com/cli/azure/install-azure-cli. 2.15 is the minimum version your need. Check the version with `az version`. You can use Azure Cloud Shell which has Azure CLI pre-installed: https://docs.microsoft.com/en-us/azure/cloud-shell/quickstart.
3. Install and configure the CLI 2.0 as documented here: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli
4. (Optional) Familiarize yourself with CLI 2.0 Jobs: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli

Steps:
1. Make sure your setup is working with either of the list commands: `az ml compute list`, `az ml jobs list`, or `az ml data list`

2. Enable private preview features.
Linux/Bash:
```
 export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED="true"
```
Windows Command Prompt:
```
set AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true
```
Windows PowerShell:
```
$Env:AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=$true
```

You may want to consider adding this command to the scripts that are run when a session is started such as [.bashrc](https://linuxize.com/post/bashrc-vs-bash-profile/) in Linux.

3. Clone this samples repo: 

```
cd $HOME
mkdir repos
cd repos
git clone https://github.com/Azure/azureml-previews
cd azureml-previews/previews/pipelines/samples
```

4. Go to the respective samples directory and try out the samples as documented in the [samples catalog](../samples/README.md). Our pick, if you have time to try only one sample: [samples/1b_e2e_registered_components](../samples/1b_e2e_registered_components)

### Known issues (to be fixed soon...)

1. New environments and datasets are created each time job is submitted even when a registered environment or dataset are specified. 
2. Cannot hard-code data or values in `component_job` in a `pipeline_job`. Need to map everything to `inputs` or `outputs` at the `pipeline_job` level.
3. Cannot skip mapping component input values that have defaults specified in the component definition.
4. Cannot use Designer UI to drag-n-drop Components and create Pipelines.
5. Input strings with spaces must use escaped quotes. E.g. "\"hello python world\""

### Installing private packages
We may occasionally ask you to try out bug fixes or specific features on private packages. This can happen if we want you validate the fixes before they get into the public packages or you cannot wait until the next public release. 

1. Remove any previous extension installations

```
az extension remove -n ml; az extension remove -n azure-cli-ml
```

2. Install the private package
```
az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/sdk-cli-v2/ml-0.0.79-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2 -y
```

3. Follow the setup instructions as documented here: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli#set-up


4. Enable private preview features as explained in Step 2 above.

