# Job Submission: Train a machine learning model

In this example you will train a light GBM model on the iris dataset and track the metrics using MLFlow. In contrast to the previous examples, this example uses a custom environment definition to run the job.

## Prerequisites

* [Ensure you have completed the setting up guidance for this repo](../../../README.md)
* [Completed the Hello World Tutorial](../hello-world/README.md)
* [Completed the Hello Data Tutorial](../hello-data/README.md)

## Training code

The 'training code' we want to submit to an AzureML compute cluster is defined in [train.py](./src/train.py). You will notice in this code that you are using MLFlow APIs to log the metrics of the light gbm model.

```Python
# start run
run = mlflow.start_run()

# enable automatic logging
mlflow.lightgbm.autolog()
```

You can run the script locally using the following in your terminal:

```Bash
cd ./tutorial/an-introduction/train-model
DATA=https://azuremlexamples.blob.core.windows.net/datasets/iris.csv
python ./src/train.py --data-path $DATA
```

## Submit to an AzureML compute cluster

To run the train.py file on a compute cluster, run the control code:

```Bash
cd ./tutorial/an-introduction/train-model
python job.py
```

## Understand the control code changes

The only difference in the control code is the environment definition, which is now done using a [pip requirements file](./environments/requirements.txt) rather than a curated environment:

```Python
env = Environment.from_pip_requirements(
    name="my_env",
    file_path="./environments/requirements.txt"
)
```

You could create environments using conda dependencies, for example:

```Python
env = Environment.from_conda_dependencies(
    name="my_env",
    file_path="./environments/conda_dependencies.yml"
)
```

Another alternative is to use a [custom docker image](https://docs.microsoft.com/azure/machine-learning/how-to-train-with-custom-image). As an example, if you wanted to use the [fast.ai image on docker hub](https://hub.docker.com/u/fastdotai), then you would specify your environment as follows:

```Python
fastai_env = Environment("fastai2")
fastai_env.docker.enabled = True
fastai_env.docker.base_image = "fastdotai/fastai2:latest"
fastai_env.python.user_managed_dependencies = True
```
