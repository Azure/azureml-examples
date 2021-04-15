# Train Models (Create Jobs)

A Job is a Resource that specifies all aspects of a computation job. It aggregates 3 things:

1. What to run
2. How to run it
3. Where to run it

A user can execute a job via the Azure Machine Learning REST API. The examples below encapsulate how a user might expand their job definition as they progress with their work.

## Set Global Variables:

```bash
export SUBSCRIPTION_ID="<your subscription ID to create the factory>"
export RESOURCE_GROUP="<your resource group to create the factory>"
export WORKSPACE="<your workspace name>"
export API_VERSION="2021-03-01-preview"
export TOKEN="<your token here>"
```

## Create your first job

For this example, we'll simply clone the v2 preview repo and run the first example!

```bash
git clone https://github.com/Azure/azureml-v2-preview
```

Check that a compute cluster exists in your workspace and you have a compute cluster named **goazurego** (if not, you can modify the name of the cluster in your YML file).

### Create the Code:

The following assumes you've uploaded `src/hello.py` to the root directory of your default container. This example uses the default datastore. We will show you how to create a new datastore in the following section. 

```bash
curl --location --request PUT 'https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/hello/versions/1?api-version=$API_VERSION' \
--header 'Authorization: Bearer $TOKEN' \
--header 'Content-Type: application/json' \
--data-raw '{
  "properties": {
    "description": "Hello World code",
    "datastoreId": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore",
    "path": "src"
  }
}'
```

### Create the Command job with the code:

```bash
curl --location --request PUT 'https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/helloWorld?api-version=$API_VERSION' \
--header 'Authorization: Bearer $TOKEN' \
--header 'Content-Type: application/json' \
--data-raw '{
    "properties": {
        "jobType": "Command",
        "command": "python hello.py",
        "environmentId": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/AzureML-Tutorial/versions/1",
        "experimentName": "helloWorld",
        "compute": {
            "target": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/goazurego",
            "instanceCount": 1
        }
    }
}'
```

This will run a simple "hello world" python script.


## Train an XGBoost model

Next, let's train an xgboost model on an IRIS dataset.

Let's navigate to the examples/iris directory in the repository and see what we should do next.

```bash
cd ./examples/iris/
```
    
### Define your environment:

First we are going to define the xgboost environment we want to run.

```bash
curl --location --request PUT 'https://management.azure.co/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/xgboost-environment/versions/1?api-version=$API_VERSION' \
--header 'Authorization: Bearer $TOKEN' \
--header 'Content-Type: application/json' \
--data-raw '{
    "properties":{
        "condaFile": "channels:\n  - conda-forge\ndependencies:\n  - python=3.6.1\n  - numpy\n  - pip\n  - pip:\n    - nbgitpuller\n    - sphinx-gallery\n    - pandas\n    - matplotlib\n    - xgboost\n    - scikit-learn\n    - azureml-mlflow",
        "Docker": {
            "type": "Image",
            "DockerImageUri": "mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20210301.v1"
        }
    }
}'
```

### Create a new Datastore
Let's create a new Datastore to house the data for this experiment. We are going to create a Datastore called `localuploads`.

```bash
curl --location --request PUT 'https://management.azure.co/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/localuploads?api-version=$API_VERSION' \
--header 'Authorization: Bearer $TOKEN' \
--header 'Content-Type: application/json' \
--data-raw '{
    "properties": {
        "contents": {
            "type": "AzureBlob",
            "accountName": "<your account name here>",
            "containerName": "azureml",
            "endpoint": "core.windows.net",
            "protocol": "https",
            "credentials": {
                "type": "AccountKey",
                "key": "<your storage key here>"
            }
        },
        "description": "My local uploads",
    }
}'
```

### Upload data to the cloud

Next the input data needs to be moved to the cloud -- therefore the user can create a data artifact in the workspace like so:

TODO: upload `data` folder to blob store and new datastore

The above command uploads the data from the local folder `data/` to the `localuploads` datastore.

### Create the Data entity:

```bash
curl --location --request PUT 'https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/data/irisdata/versions/1?api-version=$API_VERSION' \
--header 'Authorization: Bearer $TOKEN' \
--header 'Content-Type: application/json' \
--data-raw '{
  "properties": {
    "description": "Iris datset",
    "datasetType": "Simple",
    "datastoreId": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/localuploads",
    "Path": "data"
  }
}'
```

### Create the code container:

In this step, you can upload training code folder to the datastore. 

```bash
curl --location --request PUT 'https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-xgboost/versions/1?api-version=$API_VERSION' \
--header 'Authorization: Bearer $TOKEN' \
--header 'Content-Type: application/json' \
--data-raw '{
  "properties": {
    "description": "Train code",
    "datastoreId": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore",
    "path": "train"
  }
}'
```
### Create your xgboost training job
   
To submit the job:

```bash
curl --location --request PUT 'https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/xgboost?api-version=$API_VERSION' \
--header 'Authorization: Bearer $TOKEN' \
--header 'Content-Type: application/json' \
--data-raw '{
    "properties": {
        "jobType": "Command",
        "codeId": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-xgboost/versions/1",
        "command": "python train.py",
        "environmentId": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/xgboost-environment/versions/1",
        "inputDataBindings": {
            "test": {
                "dataId": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/data/irisdata/versions/1"
            }
        },
        "experimentName": "train-xgboost-job",
        "compute": {
            "target": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/goazurego",
            "instanceCount": 1
        }
    }
}'
```
  
## Sweep Jobs (Hyperparameter Tuning)

A Sweep job executes a hyperparameter sweep of a specific search space for a job. The below example uses the command job from the previous section as the 'trial' job in the sweep. It sweeps over different learning rates and subsample rates for each child run. The search space parameters will be passed as arguments to the command in the trial job.

Under `properties`, you can put the `jobType` as `Sweep` and specify the paramters for hyperparameter tuning. 

```bash
curl --location --request PUT 'https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/jobs/xgboost-sweep?api-version=$API_VERSION' \
--header 'Authorization: Bearer $TOKEN' \
--header 'Content-Type: application/json' \
--data-raw '{
    "properties": {
        "algorithm": "Random",
        "jobType": "Sweep",
        "trial":{
            "codeId": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/codes/train-xgboost/versions/1",
            "command": "python train.py",
            "environmentId": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/environments/xgboost-environment/versions/1"
        },
        "experimentName": "tune-iris-example",
        "compute": {
            "target": "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/computes/goazurego",
            "instanceCount": 1
        },
        "objective": {
            "primaryMetric": "Accuracy",
            "goal": "Maximize"
        },
        "searchSpace": {
            "--learning_rate": ["uniform", [0.001, 0.1]],
            "--subsample": ["uniform", [0.1, 1.0]]
        },
        "maxTotalTrials": 10,
        "maxConcurrentTrials": 10,
        "timeout": "PT20M"
    }
}'
```
