---
title: VS Code Snippets
---

We have compiled a collection of useful templates in the form of
[VS code snippets](https://code.visualstudio.com/docs/editor/userdefinedsnippets).

![VS Code Snippets](vs-code-snippets-demo.gif)

To add these snippets to your VS Code: `ctrl+shift+p` > Type "Configure user
snippets" > Select `python.json`. All of these snippets are available here:
[python.json](https://github.com/aminsaied/AzureML-CheatSheet/blob/master/snippets.json)

### Basic core imports
Import essential packages

**Prefix:** ['imports', 'workspace-imports-creation']
```
from azureml.core import Workspace, Experiment, Run, RunConfiguration, ComputeTarget, Environment, ScriptRunConfig$1
$0
```

### Pipeline Imports
Basic imports for pipeline

**Prefix:** pipeline-imports
```
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep$1
$0
```

### Create AML Workspace from config
Default workspace creation

**Prefix:** ['workspace-quick', 'fromconfig', 'from-config']
```
ws = Workspace.from_config()
$0
```

### Create AML Workspace from config and auth
Create workspace from config and auth

**Prefix:** workspace-from-config-auth
```
from azureml.core.authentication import InteractiveLoginAuthentication
config = {'subscription_id':'$1',
'resource_group':'$2',
'workspace_name' :'$3'}
auth = InteractiveLoginAuthentication()
ws = Workspace(**config,auth = auth)
$0
```

### Register Azure Blob Container From SAS
Register Azure Blob container to workspace via SAS

**Prefix:** ['datastore-register-blob-sas', 'reg-blob-sas']
```
ds = Datastore.register_azure_blob_container(
    workspace='$1',
    datastore_name='$2',
    container_name='$3',
    account_name='$4',
    sas_token='$5',
)
$0
```

### Create Compute Cluster with SSH
Create compute cluster with SSH enabled

**Prefix:** ['create-compute-cluster-ssh']
```
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException
ssh_public_key = '$1'
compute_config = AmlCompute.provisioning_configuration(vm_size='$4',min_nodes=$5, max_nodes=$6,admin_username='$7',admin_user_ssh_key=ssh_public_key,vm_priority='${8|lowpriority,dedicated|}',remote_login_port_public_access='Enabled')
cluster$0 = ComputeTarget.create(workspace=$9, name='$10', compute_config)
```

### AML Template Script Run Config
Template for control plane to launch script on AML

**Prefix:** ['scr', 'aml-template-script']
```
from azureml.core import Workspace, Experiment, ScriptRunConfig

# get workspace
ws = Workspace.from_config()

# get/create experiment
exp = Experiment(ws, '$1')

# set up script run configuration
config = ScriptRunConfig(
    source_directory='.',
    script='$2.py',
    #arguments=['--meaning', 42],
)

# submit script to AML
run = exp.submit(config)
print(run.get_portal_url()) # link to ml.azure.com
$0
```

### AML Template Estimator
Template for control plane to launch estimator on AML

**Prefix:** ['aml-template-estimator']
```
from azureml.core import Workspace, Experiment, ComputeTarget
from azureml.train.estimator import Estimator

# get workspace
ws = Workspace.from_config()

#  get/create experiment
exp = Experiment(ws, '$1')

# define compute target
compute_target = ComputeTarget(ws, '$2')

# set up script run configuration
config = Estimator(
    source_directory='.',
    entry_script='$3.py',
    compute_target=compute_target,
    #script_params={'--meaning': 42},
)

# submit script to AML
run = exp.submit(config)
print(run.get_portal_url()) # link to ml.azure.com
$0
```

### Environment-From-Pip
Create AML Environment from pip requirements.txt

**Prefix:** ['environment-from-pip']
```
from azureml.core import Environment
env = Environment.from_pip_requirements(
    name='$1',
    file_path='$2',
)

$0
```

### Environment-From-Conda-Spec
Create AML Environment from conda env.yml

**Prefix:** ['environment-from-conda-spec']
```
from azureml.core import Environment
env = Environment.from_conda_specification(
    name='$1',
    file_path='$2',
)

$0
```

### Environment-From-Conda-Existing
Create AML Environment from an existing Conda environment

**Prefix:** ['environment-from-conda-existing']
```
from azureml.core import Environment
env = Environment.from_existing_conda_environment(
    name='$1',
    conda_environment_name='$2',
)

$0
```

### Environment
Create AML Environment using the SDK

**Prefix:** ['environment-from-sdk']
```
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
env = Environment($1)

conda = CondaDependencies()

# add channels
conda.add_channel('$2')

# add conda packages
conda.add_conda_package('$3')

# add pip packages
conda.add_pip_package('$4')
$0
```

