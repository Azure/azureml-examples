# NYC Taxi Data Regression 
### This is an end-to-end machine learning pipeline which runs a linear regression to predict taxi fares in NYC. The pipeline is made up of components, each serving different functions, which can be registered with the workspace, versioned, and reused with various inputs and outputs. You can learn more about creating reusable components for your pipeline [here](https://github.com/Azure/azureml_run_specification/blob/master/specs/pipeline-component.md).
  * Merge Taxi Data
    * This component takes multiple taxi datasets (yellow and green) and merges/filters the data.
    * Input: Local data under samples/nyc_taxi_data_regression/data (multiple .csv files)
    * Output: Single filtered dataset (.csv)
  * Taxi Feature Engineering
    * This component creates features out of the taxi data to be used in training. 
    * Input: Filtered dataset from previous step (.csv)
    * Output: Dataset with 20+ features (.csv)
  * Train Linear Regression Model
    * This component splits the dataset into train/test sets and trains an sklearn Linear Regressor with the training set. 
    * Input: Data with feature set
    * Output: Trained model (pickle format) and data subset for test (.csv)
  * Predict Taxi Fares
    * This component uses the trained model to predict taxi fares on the test set.
    * Input: Linear regression model and test data from previous step
    * Output: Test data with predictions added as a column (.csv)
  * Score Model 
    * This component scores the model based on how accurate the predictions are in the test set. 
    * Input: Test data with predictions and model
    * Output: Report with model coefficients and evaluation scores (.txt) 


#### 1. Make sure you are in the `nyc_taxi_data_regression` directory for this sample.

#### 2. Register the Components with the AzureML workspace.

```
az ml component create --file prep.yml
az ml component create --file transform.yml
az ml component create --file train.yml
az ml component create --file predict.yml
az ml component create --file score.yml

```
If you are re-running samples, the version specified in the component yaml may already be registered. You can edit the component yaml to bump up the version or you can simply specify a new version using the command line.

```
az ml component create --file prep.yml --set version=<version_number>
az ml component create --file transform.yml --set version=<version_number>
az ml component create --file train.yml --set version=<version_number>
az ml component create --file predict.yml --set version=<version_number>
az ml component create --file score.yml --set version=<version_number>
```

#### 3. Submit the Pipeline Job. 

Make sure the version of the components you registered matches with the version defined in pipeline.yml. Also, make sure the compute cluster used in pipeline.yml is the one that is actually available in your workspace. 

Submit the Pipeline Job
```
az ml  job create --file pipeline.yml
```

You can also override the compute from the command line
```
az ml job create --file pipeline.yml --set defaults.component_job.compute.target=<your_compute>
```
Once you submit the job, you will find the URL to the Studio UI view the job graph and logs in the `interaction_endpoints` -> `Studio` section of the output. 


Sample output
```
C:\Users\shbijlan\repos\azureml-previews\previews\pipelines\samples\nyc_taxi_data_regression>az ml component create --file score.yml
Command group 'ml component' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Uploading score_src:   0%|                                                                 | 0.00/2.24k [00:00<?, ?B/s]
{
  "code": "azureml:/subscriptions/15ae9cb6-95c1-483d-a0e3-b1a1a3b06324/resourceGroups/shbijlan/providers/Microsoft.MachineLearningServices/workspaces/shbijlan/codes/74a81185-e2e5-43fc-b414-18a69bb25b26/versions/1",
  "command": "python score.py  --predictions {inputs.predictions}  --model {inputs.model}  --score_report {outputs.score_report}",
  "creation_context": {
    "created_at": "2021-07-21T23:10:33.898937+00:00",
    "created_by": "Sharmeelee Bijlani",
    "created_by_type": "User",
    "last_modified_at": "2021-07-21T23:10:33.994512+00:00"
  },
  "display_name": "Score",
  "environment": "azureml:/subscriptions/15ae9cb6-95c1-483d-a0e3-b1a1a3b06324/resourceGroups/shbijlan/providers/Microsoft.MachineLearningServices/workspaces/shbijlan/environments/AzureML-sklearn-0.24-ubuntu18.04-py37-cuda11-gpu/versions/3",
  "inputs": {
    "model": {
      "optional": false,
      "type": "path"
    },
    "predictions": {
      "optional": false,
      "type": "path"
    }
  },
  "is_deterministic": true,
  "name": "Score",
  "outputs": {
    "score_report": {
      "type": "path"
    }
  },
  "tags": {},
  "type": "command_component",
  "version": 30
}


C:\Users\shbijlan\repos\azureml-previews\previews\pipelines\samples\nyc_taxi_data_regression>az ml  job create --file pipeline.yml
Command group 'ml job' is experimental and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Custom pipeline job names are not supported yet. Please refer to the created pipeline job using the name: 73fb18f3-98a8-4677-b23b-d0b02bcb5826
{
  "compute": {
    "target": "azureml:cpu-cluster"
  },
  "creation_context": {
    "created_at": "2021-07-21T23:01:42.085184+00:00",
    "created_by": "Sharmeelee Bijlani",
    "created_by_type": "User"
  },
  "defaults": {
    "component_job": {
      "datastore": "azureml:workspaceblobstore"
    }
  },
  "experiment_name": "nyc_taxi_data_regression",
  "id": "azureml:/subscriptions/15ae9cb6-95c1-483d-a0e3-b1a1a3b06324/resourceGroups/shbijlan/providers/Microsoft.MachineLearningServices/workspaces/shbijlan/jobs/73fb18f3-98a8-4677-b23b-d0b02bcb5826",
  "inputs": {
    "pipeline_job_input": {
      "data": "azureml:7a2f5724-c2ea-45cf-820f-fd05d557d158:1",
      "mode": "mount"
    }
  },
  "interaction_endpoints": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/73fb18f3-98a8-4677-b23b-d0b02bcb5826?wsid=/subscriptions/15ae9cb6-95c1-483d-a0e3-b1a1a3b06324/resourcegroups/shbijlan/workspaces/shbijlan&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    "Tracking": {
      "endpoint": "azureml://westus.api.azureml.ms/mlflow/v1.0/subscriptions/15ae9cb6-95c1-483d-a0e3-b1a1a3b06324/resourceGroups/shbijlan/providers/Microsoft.MachineLearningServices/workspaces/shbijlan?"
    }
  },
  "jobs": {
    "predict-job": {
      "component": "azureml:Predict:13",
      "inputs": {
        "model_input": "jobs.train-job.outputs.model_output",
        "test_data": "jobs.train-job.outputs.test_data"
      },
      "outputs": {
        "predictions": "outputs.pipeline_job_predictions"
      },
      "type": "component_job"
    },
    "prep-job": {
      "component": "azureml:Prep:12",
      "inputs": {
        "raw_data": "inputs.pipeline_job_input"
      },
      "outputs": {
        "prep_data": "outputs.pipeline_job_prepped_data"
      },
      "type": "component_job"
    },
    "score-job": {
      "component": "azureml:Score:29",
      "inputs": {
        "model": "jobs.train-job.outputs.model_output",
        "predictions": "jobs.predict-job.outputs.predictions"
      },
      "outputs": {
        "score_report": "outputs.pipeline_job_score_report"
      },
      "type": "component_job"
    },
    "train-job": {
      "component": "azureml:Train:49",
      "inputs": {
        "training_data": "jobs.transform-job.outputs.transformed_data"
      },
      "outputs": {
        "model_output": "outputs.pipeline_job_trained_model",
        "test_data": "outputs.pipeline_job_test_data"
      },
      "type": "component_job"
    },
    "transform-job": {
      "component": "azureml:Transform:20",
      "inputs": {
        "clean_data": "jobs.prep-job.outputs.prep_data"
      },
      "outputs": {
        "transformed_data": "outputs.pipeline_job_transformed_data"
      },
      "type": "component_job"
    }
  },
  "name": "73fb18f3-98a8-4677-b23b-d0b02bcb5826",
  "outputs": {
    "pipeline_job_predictions": {
      "data": {
        "path": "/predictions"
      },
      "mode": "mount"
    },
    "pipeline_job_prepped_data": {
      "data": {
        "path": "/prepped_data"
      },
      "mode": "mount"
    },
    "pipeline_job_score_report": {
      "data": {
        "path": "/report"
      },
      "mode": "mount"
    },
    "pipeline_job_test_data": {
      "data": {
        "path": "/test_data"
      },
      "mode": "mount"
    },
    "pipeline_job_trained_model": {
      "data": {
        "path": "/trained-model"
      },
      "mode": "mount"
    },
    "pipeline_job_transformed_data": {
      "data": {
        "path": "/transformed_data"
      },
      "mode": "mount"
    }
  },
  "properties": {
    "azureml.git.dirty": "True",
    "azureml.parameters": "{}",
    "azureml.runsource": "azureml.PipelineRun",
    "mlflow.source.git.branch": "main",
    "mlflow.source.git.commit": "be83b0665af84b6af293873a52df41e37095ca56",
    "mlflow.source.git.repoURL": "https://github.com/Azure/azureml-previews",
    "runSource": "MFE",
    "runType": "HTTP"
  },
  "resourceGroup": "shbijlan",
  "status": "Preparing",
  "tags": {
    "azureml.pipelineComponent": "pipelinerun"
  },
  "type": "pipeline_job"
}

```


