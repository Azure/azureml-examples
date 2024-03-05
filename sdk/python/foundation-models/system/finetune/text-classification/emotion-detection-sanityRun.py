#!/usr/bin/env python
# coding: utf-8

# ## Text Classification - Emotion Detection 
# 
# This sample shows how use `text-classification` components from the `azureml` system registry to fine tune a model to detect emotions using emotion dataset. We then deploy the fine tuned model to an online endpoint for real time inference. The model is trained on tiny sample of the dataset with a small number of epochs to illustrate the fine tuning approach.
# 
# ### Training data
# We will use the [emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.
# 
# ### Model
# Models that can perform the `fill-mask` task are generally good foundation models to fine tune for `text-classification`. We will use the `bert-base-uncased` model in this notebook. If you opened this notebook from a specific model card, remember to replace the specific model name. Optionally, if you need to fine tune a model that is available on HuggingFace, but not available in `azureml` system registry, you can either [import](https://github.com/Azure/azureml-examples) the model or use the `huggingface_id` parameter instruct the components to pull the model directly from HuggingFace. 
# 
# ### Outline
# * Setup pre-requisites such as compute.
# * Pick a model to fine tune.
# * Pick and explore training data.
# * Configure the fine tuning job.
# * Run the fine tuning job.
# * Review training and evaluation metrics. 
# * Register the fine tuned model. 
# * Deploy the fine tuned model for real time inference.
# * Clean up resources. 

# ### 1. Setup pre-requisites
# * Install dependencies
# * Connect to AzureML Workspace. Learn more at [set up SDK authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?tabs=sdk). Replace  `<WORKSPACE_NAME>`, `<RESOURCE_GROUP>` and `<SUBSCRIPTION_ID>` below.
# * Connect to `azureml` system registry
# * Set an optional experiment name
# * Check or create compute. A single GPU node can have multiple GPU cards. For example, in one node of `Standard_NC24rs_v3` there are 4 NVIDIA V100 GPUs while in `Standard_NC12s_v3`, there are 2 NVIDIA V100 GPUs. Refer to the [docs](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu) for this information. The number of GPU cards per node is set in the param `gpus_per_node` below. Setting this value correctly will ensure utilization of all GPUs in the node. The recommended GPU compute SKUs can be found [here](https://learn.microsoft.com/en-us/azure/virtual-machines/ncv3-series) and [here](https://learn.microsoft.com/en-us/azure/virtual-machines/ndv2-series).

# Install dependencies by running below cell. This is not an optional step if running in a new environment.

# In[1]:


get_ipython().run_line_magic('pip', 'install azure-ai-ml')
get_ipython().run_line_magic('pip', 'install azure-identity')
get_ipython().run_line_magic('pip', 'install datasets==2.9.0')
get_ipython().run_line_magic('pip', 'install mlflow')
get_ipython().run_line_magic('pip', 'install azureml-mlflow')


# In[1]:


from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ClientSecretCredential,
)
from azure.ai.ml.entities import AmlCompute
import time

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

try:
    workspace_ml_client = MLClient.from_config(credential=credential)
except:
    workspace_ml_client = MLClient(
        credential,
        subscription_id="ed2cab61-14cc-4fb3-ac23-d72609214cfd",
        resource_group_name="training_rg",
        workspace_name="train-finetune-dev-workspace",
    )

# the models, fine tuning pipelines and environments are available in the AzureML system registry, "azureml"
registry_ml_client1 = MLClient(credential, registry_name="azureml")
#registry_ml_client2 = MLClient(credential, registry_name="azureml-preview-test1")

experiment_name = "Sanity_run_text-classification"

# generating a unique timestamp that can be used for names and versions that need to be unique
timestamp = str(int(time.time()))


# ### 2. Pick a foundation model to fine tune
# 
# Models that support `fill-mask` tasks are good candidates to fine tune for `text-classification`. You can browse these models in the Model Catalog in the AzureML Studio, filtering by the `fill-mask` task. In this example, we use the `bert-base-uncased` model. If you have opened this notebook for a different model, replace the model name and version accordingly. 
# 
# Note the model id property of the model. This will be passed as input to the fine tuning job. This is also available as the `Asset ID` field in model details page in AzureML Studio Model Catalog. 

# In[2]:


model_name = "bert-base-uncased"
foundation_model = registry_ml_client1.models.get(model_name, label="latest")
print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for fine tuning".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)


# ### 3. Create a compute to be used with the job
# 
# The finetune job works `ONLY` with `GPU` compute. The size of the compute depends on how big the model is and in most cases it becomes tricky to identify the right compute for the job. In this cell, we guide the user to select the right compute for the job.
# 
# `NOTE1` The computes listed below work with the most optimized configuration. Any changes to the configuration might lead to Cuda Out Of Memory error. In such cases, try to upgrade the compute to a bigger compute size.
# 
# `NOTE2` While selecting the compute_cluster_size below, make sure the compute is available in your resource group. If a particular compute is not available you can make a request to get access to the compute resources.

# In[3]:


import ast

if "computes_allow_list" in foundation_model.tags:
    computes_allow_list = ast.literal_eval(
        foundation_model.tags["computes_allow_list"]
    )  # convert string to python list
    print(f"Please create a compute from the above list - {computes_allow_list}")
else:
    computes_allow_list = None
    print("Computes allow list is not part of model tags")


# In[4]:


# If you have a specific compute size to work with change it here. By default we use the 1 x V100 compute from the above list
compute_cluster_size = "Standard_ND40rs_v2"

# If you already have a gpu cluster, mention it here. Else will create a new one with the name 'gpu-cluster-big'
compute_cluster = "gpu-cluster-big-new1"

try:
    compute = workspace_ml_client.compute.get(compute_cluster)
    print("The compute cluster already exists! Reusing it for the current run")
except Exception as ex:
    print(
        f"Looks like the compute cluster doesn't exist. Creating a new one with compute size {compute_cluster_size}!"
    )
    try:
        print("Attempt #1 - Trying to create a dedicated compute")
        compute = AmlCompute(
            name=compute_cluster,
            size=compute_cluster_size,
            tier="Dedicated",
            max_instances=2,  # For multi node training set this to an integer value more than 1
        )
        workspace_ml_client.compute.begin_create_or_update(compute).wait()
    except Exception as e:
        try:
            print(
                "Attempt #2 - Trying to create a low priority compute. Since this is a low priority compute, the job could get pre-empted before completion."
            )
            compute = AmlCompute(
                name=compute_cluster,
                size=compute_cluster_size,
                tier="LowPriority",
                max_instances=2,  # For multi node training set this to an integer value more than 1
            )
            workspace_ml_client.compute.begin_create_or_update(compute).wait()
        except Exception as e:
            print(e)
            raise ValueError(
                f"WARNING! Compute size {compute_cluster_size} not available in workspace"
            )


# Sanity check on the created compute
compute = workspace_ml_client.compute.get(compute_cluster)
if compute.provisioning_state.lower() == "failed":
    raise ValueError(
        f"Provisioning failed, Compute '{compute_cluster}' is in failed state. "
        f"please try creating a different compute"
    )

if computes_allow_list is not None:
    computes_allow_list_lower_case = [x.lower() for x in computes_allow_list]
    if compute.size.lower() not in computes_allow_list_lower_case:
        raise ValueError(
            f"VM size {compute.size} is not in the allow-listed computes for finetuning"
        )
else:
    # Computes with K80 GPUs are not supported
    unsupported_gpu_vm_list = [
        "standard_nc6",
        "standard_nc12",
        "standard_nc24",
        "standard_nc24r",
    ]
    if compute.size.lower() in unsupported_gpu_vm_list:
        raise ValueError(
            f"VM size {compute.size} is currently not supported for finetuning"
        )


# This is the number of GPUs in a single node of the selected 'vm_size' compute.
# Setting this to less than the number of GPUs will result in underutilized GPUs, taking longer to train.
# Setting this to more than the number of GPUs will result in an error.
gpu_count_found = False
workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
available_sku_sizes = []
for compute_sku in workspace_compute_sku_list:
    available_sku_sizes.append(compute_sku.name)
    if compute_sku.name.lower() == compute.size.lower():
        gpus_per_node = compute_sku.gpus
        gpu_count_found = True
# if gpu_count_found not found, then print an error
if gpu_count_found:
    print(f"Number of GPU's in compute {compute.size}: {gpus_per_node}")
else:
    raise ValueError(
        f"Number of GPU's in compute {compute.size} not found. Available skus are: {available_sku_sizes}."
        f"This should not happen. Please check the selected compute cluster: {compute_cluster} and try again."
    )


# ### 4. Pick the dataset for fine-tuning the model
# 
# We use the [emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset. The next few cells show basic data preparation for fine tuning:
# * Visualize some data rows
# * Replace numerical categories in data with the actual string labels. This mapping is available in the [./emotion-dataset/label.json](./emotion-dataset/label.json). This step is needed if you want string labels such as `anger`, `joy`, etc. returned when scoring the model. If you skip this step, the model will return numerical categories such as 0, 1, 2, etc. and you will have to map them to what the category represents yourself. 
# * We want this sample to run quickly, so save smaller `train`, `validation` and `test` files containing 10% of the original. This means the fine tuned model will have lower accuracy, hence it should not be put to real-world use. 
# 
# ##### Here is an example of how the data should look like
# 
# Single text classification requires the training data to include at least 2 fields – one for ‘Sentence1’ and ‘Label’ like in this example. Sentence 2 can be left blank in this case. The below examples are from Emotion dataset. 
# 
# | Text (Sentence1) | Label (Label) |
# | :- | :- |
# | i feel so blessed to be able to share it with you all | joy | 
# | i feel intimidated nervous and overwhelmed and i shake like a leaf | fear | 
# 
#  
# 
# Text pair classification, where you have two sentences to be classified (e.g., sentence entailment) will need the training data to have 3 fields – for ‘Sentence1’, ‘Sentence2’ and ‘Label’ like in this example. The below examples are from Microsoft Research Paraphrase Corpus dataset. 
# 
# | Text1 (Sentence 1) | Text2 (Sentence 2) | Label_text (Label) |
# | :- | :- | :- |
# | Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence . | Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence . | equivalent |
# | Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion . | Yucaipa bought Dominick 's in 1995 for \$ 693 million and sold it to Safeway for \$ 1.8 billion in 1998 . | not equivalent |
# 
#  

# In[5]:


# download the dataset using the helper script. This needs datasets library: https://pypi.org/project/datasets/
import os

exit_status = os.system("python ./download-dataset.py --download_dir emotion-dataset")
if exit_status != 0:
    raise Exception("Error downloading dataset")


# In[6]:


# load the ./emotion-dataset/train.jsonl file into a pandas dataframe and show the first 5 rows
import pandas as pd

pd.set_option(
    "display.max_colwidth", 0
)  # set the max column width to 0 to display the full text
df = pd.read_json("./emotion-dataset/train.jsonl", lines=True)
df.head()


# In[7]:


# load the id2label json element of the ./emotion-dataset/label.json file into pandas table with keys as 'label' column of int64 type and values as 'label_string' column as string type
import json

with open("./emotion-dataset/label.json") as f:
    id2label = json.load(f)
    id2label = id2label["id2label"]
    label_df = pd.DataFrame.from_dict(
        id2label, orient="index", columns=["label_string"]
    )
    label_df["label"] = label_df.index.astype("int64")
    label_df = label_df[["label", "label_string"]]
label_df.head()


# In[8]:


# load test.jsonl, train.jsonl and validation.jsonl form the ./emotion-dataset folder into pandas dataframes
test_df = pd.read_json("./emotion-dataset/test.jsonl", lines=True)
train_df = pd.read_json("./emotion-dataset/train.jsonl", lines=True)
validation_df = pd.read_json("./emotion-dataset/validation.jsonl", lines=True)
# join the train, validation and test dataframes with the id2label dataframe to get the label_string column
train_df = train_df.merge(label_df, on="label", how="left")
validation_df = validation_df.merge(label_df, on="label", how="left")
test_df = test_df.merge(label_df, on="label", how="left")
# show the first 5 rows of the train dataframe
train_df.head()


# In[9]:


# save 10% of the rows from the train, validation and test dataframes into files with small_ prefix in the ./emotion-dataset folder
frac = 0.1
train_df.sample(frac=frac).to_json(
    "./emotion-dataset/small_train.jsonl", orient="records", lines=True
)
validation_df.sample(frac=frac).to_json(
    "./emotion-dataset/small_validation.jsonl", orient="records", lines=True
)
test_df.sample(frac=frac).to_json(
    "./emotion-dataset/small_test.jsonl", orient="records", lines=True
)


# ### 5. Submit the fine tuning job using the the model and data as inputs
#  
# Create the job that uses the `text-classification` pipeline component. [Learn more](https://github.com/Azure/azureml-assets/blob/main/training/finetune_acft_hf_nlp/components/pipeline_components/text_classification/README.md) about all the parameters supported for fine tuning.

# Define finetune parameters
# 
# Finetune parameters can be grouped into 2 categories - training parameters, optimization parameters
# 
# Training parameters define the training aspects such as - 
# 1. the optimizer, scheduler to use
# 2. the metric to optimize the finetune
# 3. number of training steps and the batch size
# and so on
# 
# Optimization parameters help in optimizing the GPU memory and effectively using the compute resources. Below are few of the parameters that belong to this category. _The optimization parameters differs for each model and are packaged with the model to handle these variations._
# 1. enable the deepspeed, ORT and LoRA
# 2. enable mixed precision training
# 2. enable multi-node training 

# In[10]:


# Training parameters
training_parameters = dict(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    metric_for_best_model="f1_macro",
)
print(f"The following training parameters are enabled - {training_parameters}")

# Optimization parameters - As these parameters are packaged with the model itself, lets retrieve those parameters
if "model_specific_defaults" in foundation_model.tags:
    optimization_parameters = ast.literal_eval(
        foundation_model.tags["model_specific_defaults"]
    )  # convert string to python dict
else:
    optimization_parameters = dict(
        apply_lora="true", apply_deepspeed="true", apply_ort="true"
    )
#optimization_parameters['apply_lora']='false'
#optimization_parameters['apply_deepspeed']='false'
#optimization_parameters['apply_ort']='false'
print(f"The following optimizations are enabled - {optimization_parameters}")


# In[11]:


from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, PipelineComponent, Job, Component
from azure.ai.ml import PyTorchDistribution, Input

# fetch the pipeline component
pipeline_component_func = registry_ml_client1.components.get(
    name="text_classification_pipeline", label="latest"
)
#label="latest"

# define the pipeline job
@pipeline()
def create_pipeline():
    text_classification_pipeline = pipeline_component_func(
        # specify the foundation model available in the azureml system registry id identified in step #3
        mlflow_model_path=foundation_model.id,
        # huggingface_id = 'bert-base-uncased', # if you want to use a huggingface model, uncomment this line and comment the above line
        compute_model_import=compute_cluster,
        compute_preprocess=compute_cluster,
        compute_finetune=compute_cluster,
        compute_model_evaluation=compute_cluster,
        # map the dataset splits to parameters
        train_file_path=Input(
            type="uri_file", path="./emotion-dataset/small_train.jsonl"
        ),
        validation_file_path=Input(
            type="uri_file", path="./emotion-dataset/small_validation.jsonl"
        ),
        test_file_path=Input(
            type="uri_file", path="./emotion-dataset/small_test.jsonl"
        ),
        evaluation_config=Input(
            type="uri_file", path="./text-classification-config.json"
        ),
        # The following parameters map to the dataset fields
        sentence1_key="text",
        label_key="label_string",
        # Training settings
        number_of_gpu_to_use_finetuning=gpus_per_node,  # set to the number of GPUs available in the compute
        **training_parameters,
        **optimization_parameters
    )
    return {
        # map the output of the fine tuning job to the output of pipeline job so that we can easily register the fine tuned model
        # registering the model is required to deploy the model to an online or batch endpoint
        "trained_model": text_classification_pipeline.outputs.mlflow_model_folder
    }


pipeline_object = create_pipeline()

# don't use cached results from previous jobs
pipeline_object.settings.force_rerun = True

# set continue on step failure to False
pipeline_object.settings.continue_on_step_failure = False


# Submit the job

# In[12]:


# submit the pipeline job
pipeline_job = workspace_ml_client.jobs.create_or_update(
    pipeline_object, experiment_name=experiment_name
)
# wait for the pipeline job to complete
workspace_ml_client.jobs.stream(pipeline_job.name)



