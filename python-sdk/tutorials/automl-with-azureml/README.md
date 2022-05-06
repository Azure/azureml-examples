---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: Tutorials showing how to build high quality machine learning models using Azure Automated Machine Learning.
---

# Azure Machine Learning AutoML examples
1. [Automated ML Introduction](#introduction)
1. [Setup using Compute Instances](#jupyter)
1. [Setup using a Local Conda environment](#localconda)
1. [Setup using Azure Databricks](#databricks)
1. [Automated ML SDK Sample Notebooks](#samples)
1. [Documentation](#documentation)
1. [Running using python command](#pythoncommand)
1. [Troubleshooting](#troubleshooting)

<a name="introduction"></a>
# Automated ML introduction
Automated machine learning (automated ML) builds high quality machine learning models for you by automating model and hyperparameter selection. Bring a labelled dataset that you want to build a model for, automated ML will give you a high quality machine learning model that you can use for predictions.


If you are new to Data Science, automated ML will help you get jumpstarted by simplifying machine learning model building. It abstracts you from needing to perform model selection, hyperparameter selection and in one step creates a high quality trained model for you to use.

If you are an experienced data scientist, automated ML will help increase your productivity by intelligently performing the model and hyperparameter selection for your training and generates high quality models much quicker than manually specifying several combinations of the parameters and running training jobs. Automated ML provides visibility and access to all the training jobs and the performance characteristics of the models to help you further tune the pipeline if you desire.

Below are the three execution environments supported by automated ML.


 <a name="jupyter"></a>
## Setup using Compute Instances - Jupyter based notebooks from a Azure Virtual Machine

1. Open the [ML Azure portal](https://ml.azure.com)
1. Select Compute
1. Select Compute Instances
1. Click New
1. Type a Compute Name, select a Virtual Machine type and select a Virtual Machine size
1. Click Create

<a name="localconda"></a>
## Setup using a Local Conda environment

To run these notebook on your own notebook server, use these installation instructions.
The instructions below will install everything you need and then start a Jupyter notebook.

### 1. Install mini-conda from [here](https://conda.io/miniconda.html), choose 64-bit Python 3.7 or higher.
- **Note**: if you already have conda installed, you can keep using it but it should be version 4.4.10 or later (as shown by: conda -V).  If you have a previous version installed, you can update it using the command: conda update conda.
There's no need to install mini-conda specifically.

### 2. Downloading the sample notebooks
- Download the sample notebooks from [GitHub](https://github.com/Azure/azureml-examples) as zip and extract the contents to a local directory.  The automated ML sample notebooks are in the "automl-with-azureml" folder.

### 3. Setup a new conda environment
The **automl_setup** script creates a new conda environment, installs the necessary packages, configures the widget and starts a jupyter notebook. It takes the conda environment name as an optional parameter.  The default conda environment name is azure_automl.  The exact command depends on the operating system.  See the specific sections below for Windows, Mac and Linux.  It can take about 10 minutes to execute.

Packages installed by the **automl_setup** script:
    <ul><li>python</li><li>nb_conda</li><li>matplotlib</li><li>numpy</li><li>cython</li><li>urllib3</li><li>scipy</li><li>scikit-learn</li><li>pandas</li><li>tensorflow</li><li>py-xgboost</li><li>azureml-sdk</li><li>azureml-widgets</li><li>pandas-ml</li></ul>

For more details refer to the [automl_env.yml](./automl_env.yml)
## Windows
Start an **Anaconda Prompt** window, cd to the **automl-with-azureml** folder where the sample notebooks were extracted and then run:
```
automl_setup
```
## Mac
Install "Command line developer tools" if it is not already installed (you can use the command: `xcode-select --install`).

Start a Terminal windows, cd to the **how-to-use-azureml/automated-machine-learning** folder where the sample notebooks were extracted and then run:

```
bash automl_setup_mac.sh
```

## Linux
cd to the **automl-with-azureml** folder where the sample notebooks were extracted and then run:

```
bash automl_setup_linux.sh
```

### 4. Running [setup-workspace.py](../../setup-workspace.py)
- Before running any samples you next need to create a Workspace by running the [setup-workspace.py](../../setup-workspace.py) script.

### 5. Running Samples
- Please make sure you use the Python [conda env:azure_automl] kernel when trying the sample Notebooks.
- Follow the instructions in the individual notebooks to explore various features in automated ML.

### 6. Starting jupyter notebook manually
To start your Jupyter notebook manually, use:

```
conda activate azure_automl
jupyter notebook
```

or on Mac or Linux:

```
source activate azure_automl
jupyter notebook
```

 <a name="databricks"></a>
## Setup using Azure Databricks

**NOTE**: Please create your Azure Databricks cluster as v7.1 (high concurrency preferred) with **Python 3** (dropdown).
**NOTE**: You should at least have contributor access to your Azure subscription to run the notebook.
- You can find the detail Readme instructions at [GitHub](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/azure-databricks/automl).
- Download the sample notebook automl-databricks-local-01.ipynb from [GitHub](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/azure-databricks/automl) and import into the Azure databricks workspace.
- Attach the notebook to the cluster.

<a name="samples"></a>
# Automated ML SDK Sample Notebooks

## Classification
- **Classify Credit Card Fraud**
    - Dataset: [Kaggle's credit card fraud detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
      - **[Jupyter Notebook (remote run)](classification-credit-card-fraud/auto-ml-classification-credit-card-fraud.ipynb)**
          - run the experiment remotely on AML Compute cluster
          - test the performance of the best model in the local environment
      - **[Jupyter Notebook (local run)](local-run-classification-credit-card-fraud/auto-ml-classification-credit-card-fraud-local.ipynb)**
          - run experiment in the local environment
          - use Mimic Explainer for computing feature importance
          - deploy the best model along with the explainer to an Azure Kubernetes (AKS) cluster, which will compute the raw and engineered feature importances at inference time
- **Predict Term Deposit Subscriptions in a Bank**
    - Dataset: [UCI's bank marketing dataset](https://www.kaggle.com/janiobachmann/bank-marketing-dataset)
        - **[Jupyter Notebook](classification-bank-marketing-all-features/auto-ml-classification-bank-marketing-all-features.ipynb)**
          - run experiment remotely on AML Compute cluster to generate ONNX compatible models
          - view the featurization steps that were applied during training
          - view feature importance for the best model
          - download the best model in ONNX format and use it for inferencing using ONNXRuntime
          - deploy the best model in PKL format to Azure Container Instance (ACI)
- **Predict Newsgroup based on Text from News Article**
    - Dataset: [20 newsgroups text dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
        - **[Jupyter Notebook](classification-text-dnn/auto-ml-classification-text-dnn.ipynb)**
          - AutoML highlights here include using deep neural networks (DNNs) to create embedded features from text data
          - AutoML will use Bidirectional Encoder Representations from Transformers (BERT) when a GPU compute is used
          - Bidirectional Long-Short Term neural network (BiLSTM) will be utilized when a CPU compute is used, thereby optimizing the choice of DNN

## Regression
- **Predict Performance of Hardware Parts**
    - Dataset: Hardware Performance Dataset
        - **[Jupyter Notebook](regression/auto-ml-regression.ipynb)**
            - run the experiment remotely on AML Compute cluster
            - get best trained model for a different metric than the one the experiment was optimized for
            - test the performance of the best model in the local environment
        - **[Jupyter Notebook (advanced)](regression-explanation-featurization/auto-ml-regression-explanation-featurization.ipynb)**
            - run the experiment remotely on AML Compute cluster
            - customize featurization: override column purpose within the dataset, configure transformer parameters
            - get best trained model for a different metric than the one the experiment was optimized for
            - run a model explanation experiment on the remote cluster
            - deploy the model along the explainer and run online inferencing

## Time Series Forecasting
- **Forecast Energy Demand**
    - Dataset: [NYC energy demand data](http://mis.nyiso.com/public/P-58Blist.htm)
        - **[Jupyter Notebook](forecasting-energy-demand/auto-ml-forecasting-energy-demand.ipynb)**
          - run experiment remotely on AML Compute cluster
          - use lags and rolling window features
          - view the featurization steps that were applied during training
          - get the best model, use it to forecast on test data and compare the accuracy of predictions against real data
- **Forecast Orange Juice Sales (Multi-Series)**
    - Dataset: [Dominick's grocery sales of orange juice](forecasting-orange-juice-sales/dominicks_OJ.csv)
        - **[Jupyter Notebook](forecasting-orange-juice-sales/auto-ml-forecasting-orange-juice-sales.ipynb)**
          - run experiment remotely on AML Compute cluster
          - customize time-series featurization, change column purpose and override transformer hyper parameters
          - evaluate locally the performance of the generated best model
          - deploy the best model as a webservice on Azure Container Instance (ACI)
          - get online predictions from the deployed model
- **Forecast Demand of a Bike-Sharing Service**
    - Dataset: [Bike demand data](forecasting-bike-share/bike-no.csv)
        - **[Jupyter Notebook](forecasting-bike-share/auto-ml-forecasting-bike-share.ipynb)**
          - run experiment remotely on AML Compute cluster
          - integrate holiday features
          - run rolling forecast for test set that is longer than the forecast horizon
          - compute metrics on the predictions from the remote forecast
- **The Forecast Function Interface**
    - Dataset: Generated for sample purposes
        - **[Jupyter Notebook](forecasting-forecast-function/auto-ml-forecasting-function.ipynb)**
          - train a forecaster using a remote AML Compute cluster
          - capabilities of forecast function (e.g. forecast farther into the horizon)
          - generate confidence intervals
- **Forecast Beverage Production**
    - Dataset: [Monthly beer production data](forecasting-beer-remote/Beer_no_valid_split_train.csv)
        - **[Jupyter Notebook](forecasting-beer-remote/auto-ml-forecasting-beer-remote.ipynb)**
          - train using a remote AML Compute cluster
          - enable the DNN learning model
          - forecast on a remote compute cluster and compare different model performance
- **Hierarchical Time Series Forecasting**
    - Dataset: [HTS dataset](forecasting-hierarchical-timeseries/Data/hts_train.csv)
        - **[Jupyter Notebook](forecasting-hierarchical-timeseries/auto-ml-forecasting-hierarchical-timeseries.ipynb)**
          - train and forecast using a remote AML Compute cluster with multiple nodes
          - multiple AutoML runs trigger in parallel
          - data aggregation is performed at train level
- **Continuous Retraining with NOAA Weather Data**
    - Dataset: [NOAA weather data from Azure Open Datasets](https://azure.microsoft.com/en-us/services/open-datasets/)
        - **[Jupyter Notebook](continuous-retraining/auto-ml-continuous-retraining.ipynb)**
          - continuously retrain a model using Pipelines and AutoML
          - create a Pipeline to upload a time series dataset to an Azure blob
          - create a Pipeline to run an AutoML experiment and register the best resulting model in the Workspace
          - publish the training pipeline created and schedule it to run daily

## Computer Vision
- **Image Classification Multi-Class**
    - Dataset: Toy dataset with images of products found in a fridge
        - **[Jupyter Notebook](image-classification-multiclass/auto-ml-image-classification-multiclass.ipynb)**
          - train an Image Classification Multi-Class model using AutoML
          - tune hyperparameters of the model to optimize model performance
          - deploy the model to use in inference scenarios
- **Image Classification Multi-Label**
    - Dataset: Toy dataset with images of products found in a fridge
        - **[Jupyter Notebook](image-classification-multilabel/auto-ml-image-classification-multilabel.ipynb)**
          - train an Image Classification Multi-Label model using AutoML
          - tune hyperparameters of the model to optimize model performance
          - deploy the model to use in inference scenarios
- **Object Detection**
    - Dataset: Toy dataset with images of products found in a fridge
        - **[Jupyter Notebook](image-object-detection/auto-ml-image-object-detection.ipynb)**
          - train an Object Detection model using AutoML
          - tune hyperparameters of the model to optimize model performance
          - deploy the model to use in inference scenarios
- **Instance Segmentation**
    - Dataset: Toy dataset with images of products found in a fridge
        - **[Jupyter Notebook](image-instance-segmentation/auto-ml-image-instance-segmentation.ipynb)**
          - train an Instance Segmentation model using AutoML
          - tune hyperparameters of the model to optimize model performance
          - deploy the model to use in inference scenarios
- **Batch Scoring with an Image Classification Model**
    - Dataset: Toy dataset with images of products found in a fridge
        - **[Jupyter Notebook](image-classification-multiclass-batch-scoring/auto-ml-image-classification-multiclass-batch-scoring.ipynb)**
          - register an Image Classification Multi-Class model already trained using AutoML
          - create an Inference Dataset
          - provision compute targets and create a Batch Scoring script
          - use ParallelRunStep to do batch scoring
          - build, run, and publish a pipeline
          - enable a REST endpoint for the pipeline

<a name="documentation"></a>
See [Configure automated machine learning experiments](https://docs.microsoft.com/azure/machine-learning/service/how-to-configure-auto-train) to learn how more about the settings and features available for automated machine learning experiments.

<a name="pythoncommand"></a>
# Running using python command
Jupyter notebook provides a File / Download as / Python (.py) option for saving the notebook as a Python file.
You can then run this file using the python command.
However, on Windows the file needs to be modified before it can be run.
The following condition must be added to the main code in the file:

    if __name__ == "__main__":

The main code of the file must be indented so that it is under this condition.

<a name="troubleshooting"></a>
# Troubleshooting
## automl_setup fails
1. On Windows, make sure that you are running automl_setup from an Anconda Prompt window rather than a regular cmd window.  You can launch the "Anaconda Prompt" window by hitting the Start button and typing "Anaconda Prompt".  If you don't see the application "Anaconda Prompt", you might not have conda or mini conda installed.  In that case, you can install it [here](https://conda.io/miniconda.html)
2. Check that you have conda 64-bit installed rather than 32-bit.  You can check this with the command `conda info`.  The `platform` should be `win-64` for Windows or `osx-64` for Mac.
3. Check that you have conda 4.7.8 or later.  You can check the version with the command `conda -V`.  If you have a previous version installed, you can update it using the command: `conda update conda`.
4. On Linux, if the error is `gcc: error trying to exec 'cc1plus': execvp: No such file or directory`, install build essentials using the command `sudo apt-get install build-essential`.
5. Pass a new name as the first parameter to automl_setup so that it creates a new conda environment. You can view existing conda environments using `conda env list` and remove them with `conda env remove -n <environmentname>`.

## automl_setup_linux.sh fails
If automl_setup_linux.sh fails on Ubuntu Linux with the error: `unable to execute 'gcc': No such file or directory`
1. Make sure that outbound ports 53 and 80 are enabled.  On an Azure VM, you can do this from the Azure Portal by selecting the VM and clicking on Networking.
2. Run the command: `sudo apt-get update`
3. Run the command: `sudo apt-get install build-essential --fix-missing`
4. Run `automl_setup_linux.sh` again.

## Sample notebook fails
If a sample notebook fails with an error that property, method or library does not exist:
1) Check that you have selected correct kernel in jupyter notebook.  The kernel is displayed in the top right of the notebook page.  It can be changed using the `Kernel | Change Kernel` menu option.  For Azure Notebooks, it should be `Python 3.6`.  For local conda environments, it should be the conda environment name that you specified in automl_setup.  The default is azure_automl.  Note that the kernel is saved as part of the notebook.  So, if you switch to a new conda environment, you will have to select the new kernel in the notebook.
2) Check that the notebook is for the SDK version that you are using.  You can check the SDK version by executing `azureml.core.VERSION` in a jupyter notebook cell.  You can download previous version of the sample notebooks from GitHub by clicking the `Branch` button, selecting the `Tags` tab and then selecting the version.

## Numpy import fails on Windows
Some Windows environments see an error loading numpy with the latest Python version 3.6.8.  If you see this issue, try with Python version 3.6.7.

## Numpy import fails
Check the tensorflow version in the automated ml conda environment. Supported versions are < 1.13. Uninstall tensorflow from the environment if version is >= 1.13
You may check the version of tensorflow and uninstall as follows
1) start a command shell, activate conda environment where automated ml packages are installed
2) enter `pip freeze` and look for `tensorflow` , if found, the version listed should be < 1.13
3) If the listed version is a not a supported version,  `pip uninstall tensorflow` in the command shell and enter y for confirmation.

## Remote run: disk full
Automated ML creates files under /tmp/azureml_runs for each iteration that it runs.  It creates a folder with the iteration id.  For example: AutoML_9a038a18-77cc-48f1-80fb-65abdbc33abe_93.  Under this, there is a azureml-logs folder, which contains logs.  If you run too many iterations on the same DSVM, these files can fill the disk.
You can delete the files under /tmp/azureml_runs or just delete the VM and create a new one.
If your get_data downloads files, make sure the delete them or they can use disk space as well.
When using DataStore, it is good to specify an absolute path for the files so that they are downloaded just once.  If you specify a relative path, it will download a file for each iteration.

## Remote run: Iterations fail and the log contains "MemoryError"
This can be caused by insufficient memory on the DSVM.  Automated ML loads all training data into memory.  So, the available memory should be more than the training data size.
If you are using a remote DSVM, memory is needed for each concurrent iteration.  The max_concurrent_iterations setting specifies the maximum concurrent iterations.  For example, if the training data size is 8Gb and max_concurrent_iterations is set to 10, the minimum memory required is at least 80Gb.
To resolve this issue, allocate a DSVM with more memory or reduce the value specified for max_concurrent_iterations.

## Remote run: Iterations show as "Not Responding" in the RunDetails widget.
This can be caused by too many concurrent iterations for a remote DSVM.  Each concurrent iteration usually takes 100% of a core when it is running.  Some iterations can use multiple cores.  So, the max_concurrent_iterations setting should always be less than the number of cores of the DSVM.
To resolve this issue, try reducing the value specified for the max_concurrent_iterations setting.
