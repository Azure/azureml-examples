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


#### 2. Submit the Pipeline Job. 

Make sure the compute cluster used in job.yml is the one that is actually available in your workspace. 

Submit the Pipeline Job
```
az ml  job create --file pipeline.yml
```

Once you submit the job, you will find the URL to the Studio UI view the job graph and logs in the `Studio.endpoints` -> `services` section of the output. 


Sample output
```
(cliv2-dev) PS D:\azureml-examples-lochen\cli\jobs\pipelines-with-components\nyc_taxi_data_regression> az ml job create -f pipeline.yml
Command group 'ml job' is in preview and under development. Reference and support levels: https://aka.ms/CLI_refstatus
Asset labels are still in preview and may resolve to an incorrect asset version.
{
  "creation_context": {
    "created_at": "2022-03-15T11:25:38.323397+00:00",
    "created_by": "Long Chen",
    "created_by_type": "User"
  },
  "experiment_name": "nyc_taxi_data_regression",
  "id": "azureml:/subscriptions/ee85ed72-2b26-48f6-a0e8-cb5bcf98fbd9/resourceGroups/pipeline-pm/providers/Microsoft.MachineLearningServices/workspaces/pm-dev/jobs/6cef8ff4-2bd3-4101-adf2-11e0b62e6f6d",
  "inputs": {
    "pipeline_job_input": {
      "mode": "ro_mount",
      "path": "azureml:azureml://datastores/workspaceblobstore/paths/LocalUpload/aa784b6f4b0d0d3090bcd00415290f39/data",
      "type": "uri_folder"
    }
  },
  "jobs": {
    "predict-job": {
      "$schema": "{}",
      "command": "",
      "component": "azureml:49fa5eab-ad35-e3eb-27bc-5568fd2dcd74:1",
      "environment_variables": {},
      "inputs": {
        "model_input": "${{parent.jobs.train-job.outputs.model_output}}",
        "test_data": "${{parent.jobs.train-job.outputs.test_data}}"
      },
      "outputs": {
        "predictions": "${{parent.outputs.pipeline_job_predictions}}"
      },
      "type": "command"
    },
    "prep-job": {
      "$schema": "{}",
      "command": "",
      "component": "azureml:526bfb0e-aba5-36f3-ab06-2b4df9ec1554:1",
      "environment_variables": {},
      "inputs": {
        "raw_data": "${{parent.inputs.pipeline_job_input}}"
      },
      "outputs": {
        "prep_data": "${{parent.outputs.pipeline_job_prepped_data}}"
      },
      "type": "command"
    },
    "score-job": {
      "$schema": "{}",
      "command": "",
      "component": "azureml:f0ae472c-7639-1b4a-47ff-3155384584cf:1",
      "environment_variables": {},
      "inputs": {
        "model": "${{parent.jobs.train-job.outputs.model_output}}",
        "predictions": "${{parent.jobs.predict-job.outputs.predictions}}"
      },
      "outputs": {
        "score_report": "${{parent.outputs.pipeline_job_score_report}}"
      },
      "type": "command"
    },
    "train-job": {
      "$schema": "{}",
      "command": "",
      "component": "azureml:df45efbf-8373-82fd-7d5e-56fa3cd31c05:1",
      "environment_variables": {},
      "inputs": {
        "training_data": "${{parent.jobs.transform-job.outputs.transformed_data}}"
      },
      "outputs": {
        "model_output": "${{parent.outputs.pipeline_job_trained_model}}",
        "test_data": "${{parent.outputs.pipeline_job_test_data}}"
      },
      "type": "command"
    },
    "transform-job": {
      "$schema": "{}",
      "command": "",
      "component": "azureml:107ae7d3-7813-1399-34b1-17335735496c:1",
      "environment_variables": {},
      "inputs": {
        "clean_data": "${{parent.jobs.prep-job.outputs.prep_data}}"
      },
      "outputs": {
        "transformed_data": "${{parent.outputs.pipeline_job_transformed_data}}"
      },
      "type": "command"
    }
  },
  "name": "6cef8ff4-2bd3-4101-adf2-11e0b62e6f6d",
  "outputs": {
    "pipeline_job_predictions": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_prepped_data": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_score_report": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_test_data": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_trained_model": {
      "mode": "upload",
      "type": "uri_folder"
    },
    "pipeline_job_transformed_data": {
      "mode": "upload",
      "type": "uri_folder"
    }
  },
  "properties": {
    "azureml.continue_on_step_failure": "False",
    "azureml.git.dirty": "True",
    "azureml.parameters": "{}",
    "azureml.pipelineComponent": "pipelinerun",
    "azureml.runsource": "azureml.PipelineRun",
    "mlflow.source.git.branch": "march-cli-preview",
    "mlflow.source.git.commit": "8e28ab743fd680a95d71a50e456c68757669ccc7",
    "mlflow.source.git.repoURL": "https://github.com/Azure/azureml-examples.git",
    "runSource": "MFE",
    "runType": "HTTP"
  },
  "resourceGroup": "pipeline-pm",
  "services": {
    "Studio": {
      "endpoint": "https://ml.azure.com/runs/6cef8ff4-2bd3-4101-adf2-11e0b62e6f6d?wsid=/subscriptions/ee85ed72-2b26-48f6-a0e8-cb5bcf98fbd9/resourcegroups/pipeline-pm/workspaces/pm-dev&tid=72f988bf-86f1-41af-91ab-2d7cd011db47",
      "job_service_type": "Studio"
    },
    "Tracking": {
      "endpoint": "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/ee85ed72-2b26-48f6-a0e8-cb5bcf98fbd9/resourceGroups/pipeline-pm/providers/Microsoft.MachineLearningServices/workspaces/pm-dev?",
      "job_service_type": "Tracking"
    }
  },
  "settings": {
    "continue_on_step_failure": false,
    "default_compute": "cpu-cluster",
    "default_datastore": "workspaceblobstore"
  },
  "status": "Preparing",
  "tags": {
    "azureml.Designer": "true"
  },
  "type": "pipeline"
}
```


