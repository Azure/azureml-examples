---
page_type: sample
languages:
- azuresdk
- python
products:
- azure-machine-learning
description: This sample shows how to use pipeline to train model use nyc dataset.
---

#  NYC Taxi Data Regression 
This is an end-to-end machine learning pipeline which runs a linear regression to predict taxi fares in NYC. The pipeline is made up of components, each serving different functions, which can be registered with the workspace, versioned, and reused with various inputs and outputs. You can learn more about creating reusable components for your pipeline [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipelines-cli).
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
    * Output: Trained model (mlflow_model) and data subset for test (mltable)
  * Predict Taxi Fares
    * This component uses the trained model to predict taxi fares on the test set.
    * Input: Linear regression model and test data from previous step
    * Output: Test data with predictions added as a column (mltable)
  * Score Model 
    * This component scores the model based on how accurate the predictions are in the test set. 
    * Input: Test data with predictions and model
    * Output: Report with model coefficients and evaluation scores (.txt) 

Please find the sample defined in [nyc_taxi_data_regression.ipynb](nyc_taxi_data_regression.ipynb).