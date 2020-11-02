---
title: Environment
description: Guide to working with Python environments in Azure ML.
keywords:
  - environment
  - python
  - conda
  - pip
  - docker
  - environment variables
---

## Azure ML Environment
Your jobs on Azure ML are reproducible, portable and can be easily scaled up to different compute targets. With this philosophy, Azure ML heavily relies on container to encapsulate the environment where your python script and [shell commands](#advanced-shell-initialization-script) will run. For majority of use cases, the environment 
consists of a base docker image and a conda environment (including pip dependencies). For R users there is also a setting for [R CRAN packages](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.environment.rcranpackage?view=azure-ml-py) which we won't get into detail here. 

Azure ML provides the following options:
1. (Default but not so useful) If user doesn't customize an environment object when submitting their run, Azure ML will use a default container image, with only one python package called `azureml-defaults` which includes only Azure ML  essentials. 
2. Use one of the [curated environment](https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments).
3. Use one of the [default base image](https://github.com/Azure/AzureML-Containers), and ask Azure ML to manage Conda dependencies by [providing a `CondaDependencies` object](#create-conda-dependencies). 
4. Use a previously [registered environment](#registered-environments).
5. Use a [custom docker image or dockerfile](#advanced-custom-docker-images). User can either use a python environment in the image directly by using this as the docker base image for the environment and set `user_managed_dependencies=True`, in this case Azure ML won't be able to manage and add extra python dependencies. Or user can ask Azure ML to manage Conda dependencies by [providing a `CondaDependencies` object](#create-conda-dependencies). 

:::tip
When the conda dependencies are managed by Azure ML (`user_managed_dependencies=False`, by default), Azure ML will check whether the same environment has already been materialized into a docker image in the Azure Container Registry associated with the Azure ML workspace. If it is a new environment, Azure ML will have a job preparation stage to build
a new docker image for the new environment. user can see a image build log file in the logs and monitor the image build progress. The job won't start until the image is built and pushed to the container registry. 

This image building process can take some time and delay your job start. To avoid unnecessary image building, consider
1. Register an environment that contains most packages you need and reuse when possible.
2. If you only need a few extra packages on top of an existing environment, 
    1. If the existing environment is a docker image, use a dockerfile from this docker image so you only need to add one layer to install a few extra packagers. 
    2. Install extra python packages in your user script so the package installation happens in the script run as part of your code instead of asking Azure ML to treat them as part of a new environment. Consider using a [setup script](#advanced-shell-initialization-script).
:::

:::info
Due to intricacy of the python package dependencies and potential version conflict, we recommend users to understand the [image building process](#how-azure-ml-build-image-from-a-environment) and use custom docker image and dockerfiles (based on Azure ML base images) to manage your own python environment. This practice not only gives users full transparency of the environment, but also saves image building times at agile development stage. 
:::


## Create Conda Dependencies

Easily create, maintain and share Python environments with **pip** and **Conda**, or directly from the **Python SDK**.

### From pip

Create Environment from pip `requirements.txt` file

```python
from azureml.core import Environment
env = Environment.from_pip_requirements('<environment-name>', '<path/to/requirements.txt>')
```

### From Conda

Create Environment from Conda `env.yml` file

```python
from azureml.core import Environment
env = Environment.from_conda_specifications('<environment-name>', '<path/to/env.yml>')
```

### From SDK

Use the `CondaDependencies` class to create a Python environment in code:

```python
from azureml.core.conda_dependencies import CondaDependencies

conda = CondaDependencies()

# add channels
conda.add_channel('pytorch')

# add conda packages
conda.add_conda_package('python=3.7')
conda.add_conda_package('pytorch')
conda.add_conda_package('torchvision')

# add pip packages
conda.add_pip_package('pyyaml')
```

Which can be consumed by an environment as follows.

```python
from azureml.core import Environment
env = Environment('pytorch')
env.python.conda_dependencies = conda
```

Converting the conda_dependencies to an `env.yml` file later is easy:

```python
conda.save('env.yml')
```

This example will generate the following file:

```yml title="env.yml"
# Conda environment specification. The dependencies defined in this file will
# be automatically provisioned for runs with userManagedDependencies=False.

# Details about the Conda environment file format:
# https://conda.io/docs/user-guide/tasks/manage-environments.html#create-env-file-manually

name: project_environment
dependencies:
  # The python interpreter version.
  # Currently Azure ML only supports 3.5.2 and later.
- python=3.7

- pip:
    # Required packages for Azure ML execution, history, and data preparation.
  - azureml-defaults

  - pyyaml
- pytorch
- torchvision
channels:
- anaconda
- conda-forge
- pytorch
```

## Registered Environments

Register an environment `env: Environment` to your workspace `ws` to reuse/share with your team.

```python
env.register(ws)
```

To see the registered Environments already available:

```python
envs: Dict[str, Environment] = ws.environments

for name, env in envs.items():
    print(name)
# Azure ML-Chainer-5.1.0-GPU
# Azure ML-Scikit-learn-0.20.3
# Azure ML-PyTorch-1.1-GPU
# ...
```

This list contains custom environments that have been registered to the workspace as well as a
collection of _curated environments_ maintained by the Azure ML team.

List the conda dependencies for a given environment, for example in 'Azure ML-Chainer-5.1.0-GPU':

```python
env = ws.environments['Azure ML-PyTorch-1.1-GPU']
print(env.python.conda_dependencies.serialize_to_string())
```

Which returns the following.

```yaml title="Azure ML-PyTorch-1.1-GPU Conda Dependencies"
channels:
- conda-forge
dependencies:
- python=3.6.2
- pip:
  - azureml-core==1.15.0
  - azureml-defaults==1.15.0
  - azureml-telemetry==1.15.0
  - azureml-train-restclients-hyperdrive==1.15.0
  - azureml-train-core==1.15.0
  - torch==1.1
  - torchvision==0.2.1
  - mkl==2018.0.3
  - horovod==0.16.1
  - tensorboard==1.14.0
  - future==0.17.1
name: azureml_eb61e39e20e87ad998ae2c88df1dd0af
```

## Save / Load Environments

Save an environment to a local directory:

```python
env.save_to_directory('<path/to/local/directory>', overwrite=True)
```

This will generate a directory with two (human-understandable and editable) files:

- `azureml_environment.json` : Metadata including name, version, environment variables and Python and Docker configuration
- `conda_dependencies.yml` : Standard conda dependencies YAML (for more deatils see [Conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)).

Load this environment later with

```python
env = Environment.load_from_directory('<path/to/local/directory>')
```

## How Azure ML Build Image from a Environment

This section explains how Azure ML builds its docker image based on an `Environment`.

Consider the following example `env.yml` file.

```yml title="env.yml"
name: pytorch
channels:
    - defaults
    - pytorch
dependencies:
    - python=3.7
    - pytorch
    - torchvision
```

Create and register this as an `Environment` in your workspace `ws`.

```python
from azureml.core import Environment
env = Environment.from_conda_specification('pytorch', 'env.yml')
env.register(ws)
```

In order to consume this environment, Azure ML builds a corresponding docker image. This dockerfile
is available as part of the `Environment` object `env`:

```python
details = env.get_image_details(ws)
print(details.dockerfile)
```

Which looks like this:

```docker title="Dockerfile" {1,7-12}
FROM mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20200821.v1@sha256:8cee6f674276dddb23068d2710da7f7f95b119412cc482675ac79ba45a4acf99
USER root
RUN mkdir -p $HOME/.cache
WORKDIR /
COPY azureml-environment-setup/99brokenproxy /etc/apt/apt.conf.d/
RUN if dpkg --compare-versions `conda --version | grep -oE '[^ ]+$'` lt 4.4.11; then conda install conda==4.4.11; fi
COPY azureml-environment-setup/mutated_conda_dependencies.yml azureml-environment-setup/mutated_conda_dependencies.yml
RUN ldconfig /usr/local/cuda/lib64/stubs && conda env create -p /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6 -
f azureml-environment-setup/mutated_conda_dependencies.yml && rm -rf "$HOME/.cache/pip" && conda clean -aqy && CONDA_ROO
T_DIR=$(conda info --root) && rm -rf "$CONDA_ROOT_DIR/pkgs" && find "$CONDA_ROOT_DIR" -type d -name __pycache__ -exec rm
 -rf {} + && ldconfig
# Azure ML Conda environment name: azureml_7459a71437df47401c6a369f49fbbdb6
ENV PATH /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6/bin:$PATH
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6
ENV LD_LIBRARY_PATH /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6/lib:$LD_LIBRARY_PATH
COPY azureml-environment-setup/spark_cache.py azureml-environment-setup/log4j.properties /azureml-environment-setup/
RUN if [ $SPARK_HOME ]; then /bin/bash -c '$SPARK_HOME/bin/spark-submit  /azureml-environment-setup/spark_cache.py'; fi
ENV AZUREML_ENVIRONMENT_IMAGE True
CMD ["bash"]
```

Notice:

- The base image here is a standard image maintained by Azure ML. Dockerfiles for all base images are available on
github: https://github.com/Azure/AzureML-Containers . You can also use your own docker image as base image. 
- The dockerfile references `mutated_conda_dependencies.yml` to build the Python environment via Conda. 

Get the contents of `mutated_conda_dependencies.yml` from the environment:

```python
env.python.conda_dependencies.serialize_to_string()
```

Which looks like

```bash title="mutated_conda_dependencies.yml"
channels:
    - defaults
    - pytorch
dependencies:
    - python=3.7
    - pytorch
    - torchvision
name: azureml_7459a71437df47401c6a369f49fbbdb6
```

If you have docker installed locally, you can build the docker image from Azure ML environment locally with option to push the image to workspace ACR directly. This is recommended when users are iterating on the dockerfile since local build can 
utilize cached layers. 
```python
build = env.build_local(workspace=ws, useDocker=True, pushImageToWorkspaceAcr=True)
```

## (Advanced) Custom Docker Images

By default, Azure ML will create your Python environment inside a Docker image it maintains.

:::info No secrets
This default image is not a secret. For example, you can see the Dockerfile used to create
it with the following:

```python
from azureml.core import Environment
env = Environment('no-secrets')             # create new Environment
env.register(ws)                            # register to the workspace
details = env.get_image_details(ws)
print(details['ingredients']['dockerfile'])
```

Dockerfiles for all base images are available on github: https://github.com/Azure/AzureML-Containers
:::

You may chose to use your own Docker image. In this case there are two options for python environment:

- Ask Azure ML to manage a new conda environment custom base docker image provided
- Use a python environment already exists in the custom base docker image provided

### Requirements for custom image

We strongly recommend user to build their custom image from one of the Azure ML base images. If user wants to build from scratch, here are a list of requirements and recommendations to keep in mind:
- **Conda**: Azure ML uses Conda to manage python environments by default. If you intent to allow Azure ML to manage the python environment, Conda is required. 
- **libfuse**: Required when using `Dataset`
- **Openmpi**: Required for distributed runs
- **nvidia/cuda**: (Recommended) For GPU-based training build image from [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda)
- **Mellanox OFED user space drivers** (Recommend) For SKUs with Infiniband 

We suggest users to look at the [dockerfiles of Azure ML base images](https://github.com/Azure/AzureML-Containers) as references.  

### Provide Python packages to the custom image

In this case we will use pip, Conda or the SDK to manage our Python packages as above, resulting
in `env: Environment`. For example,

```python
env = Environment.from_pip_requirements('nlp', 'requirements.txt')
```

Assuming you have a Dockerfile to hand you can specify the following:

```python
# just as an example
env.docker.base_image = None                    # translation: do not use your default base image
env.docker.base_dockerfile = "./Dockerfile"     # translation: use my Dockerfile as base instead
```

When you use this environment in a compute target it will build a Docker image as follows:

```docker
###
Contents of your base dockerfile
###

###
Build the Python dependencies as specified in env object
###
```

:::info
Again, you can see the exact contents of this Dockerfile used by running

```python
details = env.get_image_details(ws)
print(details['ingredients']['dockerfile'])
```
:::

### Use Python interpreter from the custom image

Usually your custom Docker image has its own Python environment already set up.

```docker title="Dockerfile"
FROM mcr.microsoft.com/azureml/base:intelmpi2018.3-ubuntu16.04
RUN conda update -n base -c defaults conda
RUN [ "/bin/bash", "-c", "conda create -n pytorch Python=3.6.2 && source activate amlbert && conda install pip"]
RUN /opt/miniconda/envs/pytorch/bin/pip install pytorch
```
In this case you need to:

- Indicate that you are managing your own Python dependencies: `user_managed_dependencies=True`
- Specify the path to your Python interpreter: `interpreter_path=<path>`

```python
env = Environment('pytorch')    # create an Environment called 'pytorch'

# set up custom docker image or a dockerfile
# env.docker.base_image = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"
# or set up base image from a dockerfile
env.docker.base_dockerfile = "./Dockerfile"

# indicate how to run Python
env.python.user_managed_dependencies=True
env.python.interpreter_path = "/opt/miniconda/bin/python"
```
### Use custom image from a private registry

Azure ML Environment can use a Custom image from a private registry as long as login information are provided. 

```python
env = Environment('myenv') # create an Environment called 'myenv'
env.docker.base_image = "/my/private/img:tag",  #image repository path
env.docker.base_image_registry.address = "myprivateacr.azurecr.io"  # private registry
# Retrieve username and password from the workspace key vault
env.docker.base_image_registry.username = ws.get_default_keyvault().get_secret("username")  
env.docker.base_image_registry.password = ws.get_default_keyvault().get_secret("password")
```


## (Advanced) Environment Variables

To set environment variables use the `environment_variables: Dict[str, str]` attribute. Environment variables
are set on the process where the user script is executed.

```python
env = Environment('example')
env.environment_variables['EXAMPLE_ENV_VAR'] = 'EXAMPLE_VALUE'
```

## (Advanced) Shell Initialization Script

A useful pattern is to run shell scripts on Azure ML compute to prepare the nodes.

In this example we show how to use initialization shell scripts for both **individual nodes** as well
as **each process**:

- `setup.sh`: This will run only on local_rank 0 process (i.e., once per node)
  - Run a utility script `download_data.py` to download training data to the node
- `run.sh` : This will run on each process
  - 

These scripts will run ahead of our main python call to `train.py`.

```
src/
  setup.sh              # init local_rank 0
  run.sh                # init for each rank
  requirements.txt      # pip requirements
  download_data.py      # utility script to download training data
  train.py              # model training script
  aml_wrapper.py        # thin wrapper calling setup.sh and run.sh
```

```bash title="setup.sh"
pip install -r requirements.txt
python download_data.py --output_dir /tmp/data
```

This script runs `download_data.py` which downloads training data to the specified output
directory `/tmp/data`.

In this example the data should be downloaded once per node in the compute cluster (not once
per process!). 

```bash title="run.sh"
python train.py --training_data /tmp/data --learning_rate 1e-6
```

This is the main call to the training script and needs to be called by each process. The data
downloaded by `download_data.py` is referenced as a command-line argument.

Finally, prepare a wrapper script to execute the above. Notice the wrapper script takes a great deal of care to make sure `setup.sh` only
executed once in each node and when there are multiple processes per node other nodes will wait when `setup.sh` is executing. A marker file is used
to mimic a barrier so all processes are in sync.  

```python title="aml_wrapper.py"
#!/usr/bin/env python
import os
import pathlib
import sys
import time

MARKER = pathlib.Path("/tmp/.aml_setup_done")

def run_command(*files, verbose=False):
  lines = []
  for file in files:
    if not os.path.exists(file):
      print("No file %s", file)
      return 1

    with open(file, 'rt') as f:
      script = f.read()
      script = script.replace('\r', '')  # for Windows submissions
      lines.extend(script.split('\n'))

  print("Executing", *files)
  if verbose:
    lines.insert(0, "set -o xtrace")
  return os.system(";".join(lines))

if __name__ == "__main__":

  if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
    if os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] == "0":
      try:
        run_command("./setup.sh", verbose=True)
      finally:
        MARKER.touch(exist_ok=True)
    while not MARKER.exists():
      time.sleep(1)

    sys.exit(run_command("./run.sh", verbose=False) >> 8)

  return_code = run_command("./setup.sh", "./run.sh")
  sys.exit(return_code >> 8)
```

Submit this to a `ComputeTarget` with `ScriptRunConfig`.

```python
from azureml.core import Workspace, ComputeTarget, ScriptRunConfig

ws = Workspace.from_config()
compute_target = ws.compute_targets['<compute-target-name>']

config = ScriptRunConfig(
    source_directory='src',
    script='aml_wrapper.py',
    compute_target=compute_target,
)
```