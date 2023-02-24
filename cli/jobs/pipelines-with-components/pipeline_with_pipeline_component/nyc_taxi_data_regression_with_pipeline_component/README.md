# NYC Taxi Data Regression 
### This is an end-to-end machine learning pipeline which runs a linear regression to predict taxi fares in NYC. The pipeline is made up of components, each serving different functions, which can be registered with the workspace, versioned, and reused with various inputs and outputs. You can learn more about creating reusable components for your pipeline [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipelines-cli).
In this example, we have two pipeline components, then we can have parent pipeline to orchestrate these pipeline components, using this approach you can easily iterate sub tasks of complex projects.
  * Data_pipeline, include data prep and transformation
      * Merge Taxi Data
        * This component takes multiple taxi datasets (yellow and green) and merges/filters the data.
        * Input: Local data under samples/nyc_taxi_data_regression/data (multiple .csv files)
        * Output: Single filtered dataset (.csv)
      * Taxi Feature Engineering
        * This component creates features out of the taxi data to be used in training. 
        * Input: Filtered dataset from previous step (.csv)
        * Output: Dataset with 20+ features (.csv)
  * Train pipeline, include train, predict and score
      * Train Linear Regression Model
        * This component splits the dataset into train/test sets and trains an sklearn Linear Regressor with the training set. 
        * Input: Data with feature set
        * Output: Trained model (mlflow_model) and data subset for test (mltable)
      * Predict Taxi Fares
        * This component uses the trained model to predict taxi fares on the test set.
        * Input: Linear regression model and test data from previous step
        * Output: Test data with predictions added as a column (mltable)
      * Score Model 
        * This component scores the model based on how accurate the predictions are in the test set. 
        * Input: Test data with predictions and model
        * Output: Report with model coefficients and evaluation scores (.txt) 
    

#### 1. Make sure you are in the `nyc_taxi_data_regression_with_pipeline_component` directory for this sample.


#### 2. Submit the Pipeline Job. 

Make sure the compute cluster used in job.yml is the one that is actually available in your workspace. 

Submit the Pipeline Job
```
az ml  job create --file pipeline.yml
```

Once you submit the job, you will find the URL to the Studio UI view the job graph and logs in the `Studio.endpoints` -> `services` section of the output. 