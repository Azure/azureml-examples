# An introduction to job submission: Hello World

This is the obligatory "Hello, World!" example. 

## Prerequisites

* [Ensure you have completed the setting up guidance for this repo](../../../README.md)

## Submit "Hello, World!" to a compute cluster

The 'training code' we want to submit to an AzureML compute cluster is defined in `./src/hello.py` and does the following:

```Python
# ./src/hello.py
print("Hello, World!")
```

To submit this as a job to Azure Machine Learning, use the following in your terminal:

```Bash
cd ./tutorials/an-introduction/hello-world
python job.py
```

The logs of the job will start streaming to your terminal. At the beginning of the stream you will also see "Link to Azure Machine Learning Portal" and a URL. If you select the URL it will take you to job in Azure Machine Learning Studio.

## Understand how AzureML executes a job
AzureML does the following for you when you submit a job:

1. If you have defined a custom environment, then a container is built in the cloud for you automatically (using Azure Container Registry). In the case of this Hello World example, you are using a *curated environment* where the container is already built for you, meaning this step is skipped (saving time).
1. Provision compute. The `cpu-cluster` we are using for this job has 0 nodes, and therefore AzureML automatically instructs the compute layer to spin up a node to run the job on. The cluster will scale back to zero when it has been idle for 20-minutes.
1. Queue your job, and when resources are available:
1. Pull container to compute.
1. Mount datasets to the compute. In this hello world example you are not using any data, so this step is skipped.
1. Run Python job.
1. Unmount datasets from compute.
1. Stop container.

With this in mind, the first time you run a job it can take around 3-5 minutes to run since creating and pulling containers adds latency. However, on subsequent runs the latency is around 15-seconds since the container is cached on the compute nodes.

## Understanding the control code 

### Imports
The code imports the AzureML SDK elements required to run a job.

```Python
from azureml.core import Workspace, Experiment, 
from azureml.core import ScriptRunConfig, Environment
```

### Connect to the AzureML workspace
[Workspace](https://docs.microsoft.com/python/api/azureml-core/azureml.core.workspace.workspace?preserve-view=true&view=azure-ml-py) connects to your Azure Machine Learning workspace, so that you can communicate with your Azure Machine Learning resources.

```Python
ws = Workspace.from_config()
```

### Define an Experiment
[Experiment](https://docs.microsoft.com/python/api/azureml-core/azureml.core.experiment.experiment?preserve-view=true&view=azure-ml-py) provides a simple way to organize multiple runs under a single name. Later you can see how experiments make it easy to compare metrics between dozens of runs.

```Python
exp = Experiment(
    workspace=ws, 
    name="an-introduction-hello-world-tutorial"
)
```

### Define an Environment
Azure Machine Learning provides the concept of an [environment](https://docs.microsoft.com/python/api/azureml-core/azureml.core.environment.environment?preserve-view=true&view=azure-ml-py) to represent a reproducible, versioned Python environment for running experiments. It's easy to create an environment from a local Conda or pip environment. In the Hello World example we are using a curated environment called *AzureML-Tutorial*, which is a pre-built container containing [these Python libraries](https://docs.microsoft.com/azure/machine-learning/resource-curated-environments#azureml-tutorial).

```Python
env = Environment.get(ws, "AzureML-Tutorial")
```

### Define the script run configuration
[ScriptRunConfig](https://docs.microsoft.com/python/api/azureml-core/azureml.core.scriptrunconfig?preserve-view=true&view=azure-ml-py) wraps your hello.py code and passes it to your workspace. As the name suggests, you can use this class to configure how you want your script to run in Azure Machine Learning. It also specifies what compute target the script will run on. In this code, the target is the compute cluster that you created in the setup tutorial.

```Python
src = ScriptRunConfig(
    source_directory="src",
    script="hello.py", 
    compute_target="cpu-cluster",
    environment=env
)
```

### Submit the job
Submits your script. This submission is called a [run](https://docs.microsoft.com/python/api/azureml-core/azureml.core.run%28class%29?preserve-view=true&view=azure-ml-py). A run encapsulates a single execution of your code. Use a run to monitor the script progress, capture the output, analyze the results, visualize metrics, and more.

```Python
run = exp.submit(src)
run.wait_for_completion(show_output=True)
```

