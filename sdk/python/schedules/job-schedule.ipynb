{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Working with Schedule\n",
    "\n",
    "**Requirements** - In order to benefit from this tutorial, you will need:\n",
    "- A basic understanding of Machine Learning\n",
    "- An Azure account with an active subscription. [Create an account for free](https://azure.microsoft.com/free/?WT.mc_id=A261C142F)\n",
    "- An Azure ML workspace - [Configure workspace](../../jobs/configuration.ipynb) \n",
    "- A python environment\n",
    "- Installed Azure Machine Learning Python SDK v2 - [install instructions](../../README.md) - check the getting started section\n",
    "\n",
    "**Learning Objectives** - By the end of this tutorial, you should be able to:\n",
    "- Create schedule from Python SDK\n",
    "- Enable/Disable schedule from Python SDK\n",
    "- Update schedule's trigger from Python SDK\n",
    "\n",
    "**Motivations** - This notebook covers the main scenario that user management schedule for job in SDK v2."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Connect to Azure Machine Learning Workspace\n",
    "\n",
    "The [workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace) is the top-level resource for Azure Machine Learning, providing a centralized place to work with all the artifacts you create when you use Azure Machine Learning. In this section we will connect to the workspace in which the job will be run.\n",
    "\n",
    "## 1.1. Import the required libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\n",
    "from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential\n",
    "\n",
    "from azure.ai.ml import MLClient, Input, load_component\n",
    "from azure.ai.ml.constants import TimeZone\n",
    "from azure.ai.ml.dsl import pipeline\n",
    "from azure.ai.ml.entities import (\n",
    "    JobSchedule,\n",
    "    CronTrigger,\n",
    "    RecurrenceTrigger,\n",
    "    RecurrencePattern,\n",
    ")\n",
    "from datetime import datetime"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.2. Configure credential\n",
    "\n",
    "We are using `DefaultAzureCredential` to get access to workspace. When an access token is needed, it requests one using multiple identities(`EnvironmentCredential, ManagedIdentityCredential, SharedTokenCacheCredential, VisualStudioCodeCredential, AzureCliCredential, AzurePowerShellCredential`) in turn, stopping when one provides a token.\n",
    "Reference [here](https://docs.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) for more information.\n",
    "\n",
    "`DefaultAzureCredential` should be capable of handling most Azure SDK authentication scenarios. \n",
    "Reference [here](https://docs.microsoft.com/en-us/python/api/azure-identity/azure.identity?view=azure-python) for all available credentials if it does not work for you.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    credential = DefaultAzureCredential()\n",
    "    # Check if given credential can get token successfully.\n",
    "    credential.get_token(\"https://management.azure.com/.default\")\n",
    "except Exception as ex:\n",
    "    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work\n",
    "    credential = InteractiveBrowserCredential()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.3. Configure workspace details and get a handle to the workspace\n",
    "\n",
    "To connect to a workspace, we need identifier parameters - a subscription, resource group and workspace name. We will use these details in the `MLClient` from `azure.ai.ml` to get a handle to the required Azure Machine Learning workspace."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    ml_client = MLClient.from_config(credential=credential)\n",
    "except Exception as ex:\n",
    "    # enter details of your AML workspace\n",
    "    subscription_id = \"<SUBSCRIPTION_ID>\"\n",
    "    resource_group = \"<RESOURCE_GROUP>\"\n",
    "    workspace = \"<AML_WORKSPACE_NAME>\"\n",
    "\n",
    "    # get a handle to the workspace\n",
    "    ml_client = MLClient(credential, subscription_id, resource_group, workspace)\n",
    "print(ml_client)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.4 Create a pipeline job with settings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "create_pipeline_job"
   },
   "outputs": [],
   "source": [
    "parent_dir = \"../jobs/pipelines/1a_pipeline_with_components_from_yaml\"\n",
    "train_model = load_component(source=parent_dir + \"/train_model.yml\")\n",
    "score_data = load_component(source=parent_dir + \"/score_data.yml\")\n",
    "eval_model = load_component(source=parent_dir + \"/eval_model.yml\")\n",
    "\n",
    "# Construct pipeline\n",
    "@pipeline()\n",
    "def pipeline_with_components_from_yaml(\n",
    "    training_input,\n",
    "    test_input,\n",
    "    training_max_epochs=20,\n",
    "    training_learning_rate=1.8,\n",
    "    learning_rate_schedule=\"time-based\",\n",
    "):\n",
    "    \"\"\"E2E dummy train-score-eval pipeline with components defined via yaml.\"\"\"\n",
    "    # Call component obj as function: apply given inputs & parameters to create a node in pipeline\n",
    "    train_with_sample_data = train_model(\n",
    "        training_data=training_input,\n",
    "        max_epochs=training_max_epochs,\n",
    "        learning_rate=training_learning_rate,\n",
    "        learning_rate_schedule=learning_rate_schedule,\n",
    "    )\n",
    "\n",
    "    score_with_sample_data = score_data(\n",
    "        model_input=train_with_sample_data.outputs.model_output, test_data=test_input\n",
    "    )\n",
    "    score_with_sample_data.outputs.score_output.mode = \"upload\"\n",
    "\n",
    "    eval_with_sample_data = eval_model(\n",
    "        scoring_result=score_with_sample_data.outputs.score_output\n",
    "    )\n",
    "\n",
    "    # Return: pipeline outputs\n",
    "    return {\n",
    "        \"trained_model\": train_with_sample_data.outputs.model_output,\n",
    "        \"scored_data\": score_with_sample_data.outputs.score_output,\n",
    "        \"evaluation_report\": eval_with_sample_data.outputs.eval_output,\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "change_run_settings"
   },
   "outputs": [],
   "source": [
    "# set run time settings\n",
    "pipeline_job = pipeline_with_components_from_yaml(\n",
    "    training_input=Input(type=\"uri_folder\", path=parent_dir + \"/data/\"),\n",
    "    test_input=Input(type=\"uri_folder\", path=parent_dir + \"/data/\"),\n",
    "    training_max_epochs=20,\n",
    "    training_learning_rate=1.8,\n",
    "    learning_rate_schedule=\"time-based\",\n",
    ")\n",
    "\n",
    "# set pipeline level compute\n",
    "pipeline_job.settings.default_compute = \"cpu-cluster\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.1 Create schedule\n",
    "### 2.1.1 Define schedule with with recurrence pattern"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "create_schedule_recurrence"
   },
   "outputs": [],
   "source": [
    "schedule_name = \"simple_sdk_create_schedule_recurrence\"\n",
    "\n",
    "schedule_start_time = datetime.utcnow()\n",
    "recurrence_trigger = RecurrenceTrigger(\n",
    "    frequency=\"day\",\n",
    "    interval=1,\n",
    "    schedule=RecurrencePattern(hours=10, minutes=[0, 1]),\n",
    "    start_time=schedule_start_time,\n",
    "    time_zone=TimeZone.UTC,\n",
    ")\n",
    "\n",
    "job_schedule = JobSchedule(\n",
    "    name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1.2 Define schedule with with cron expression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "create_schedule_cron"
   },
   "outputs": [],
   "source": [
    "schedule_name = \"simple_sdk_create_schedule_cron\"\n",
    "\n",
    "schedule_start_time = datetime.utcnow()\n",
    "cron_trigger = CronTrigger(\n",
    "    expression=\"15 10 * * *\",\n",
    "    start_time=schedule_start_time,  # start time\n",
    "    time_zone=\"Eastern Standard Time\",  # time zone of expression\n",
    ")\n",
    "\n",
    "job_schedule = JobSchedule(\n",
    "    name=schedule_name, trigger=cron_trigger, create_job=pipeline_job\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1.3 Create schedule"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "create_schedule"
   },
   "outputs": [],
   "source": [
    "job_schedule = ml_client.schedules.begin_create_or_update(\n",
    "    schedule=job_schedule\n",
    ").result()\n",
    "print(job_schedule)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.2 Disable the schedule"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "disable_schedule"
   },
   "outputs": [],
   "source": [
    "job_schedule = ml_client.schedules.begin_disable(name=schedule_name).result()\n",
    "job_schedule.is_enabled"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.3 Check the detail of the schedule"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "show_schedule"
   },
   "outputs": [],
   "source": [
    "created_schedule = ml_client.schedules.get(name=schedule_name)\n",
    "[created_schedule.name]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.4 List schedules in a workspace"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "list_schedule"
   },
   "outputs": [],
   "source": [
    "schedules = ml_client.schedules.list()\n",
    "[s.name for s in schedules]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.5 Enable a schedule"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "job_schedule = ml_client.schedules.begin_enable(name=schedule_name).result()\n",
    "job_schedule.is_enabled"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.6 Update a schedule"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "enable_schedule"
   },
   "outputs": [],
   "source": [
    "# Update trigger expression\n",
    "job_schedule.trigger.expression = \"10 10 * * 1\"\n",
    "job_schedule = ml_client.schedules.begin_create_or_update(\n",
    "    schedule=job_schedule\n",
    ").result()\n",
    "print(job_schedule)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.7 Delete the schedule"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "name": "delete_schedule"
   },
   "outputs": [],
   "source": [
    "# Only disabled schedules can be deleted\n",
    "ml_client.schedules.begin_disable(name=schedule_name).result()\n",
    "ml_client.schedules.begin_delete(name=schedule_name).result()"
   ]
  }
 ],
 "metadata": {
  "description": {
   "description": "Create a component asset"
  },
  "kernelspec": {
   "display_name": "Python 3.10 - SDK V2",
   "language": "python",
   "name": "python310-sdkv2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.4"
  },
  "vscode": {
   "interpreter": {
    "hash": "0d334cf3c07019271c164ebdcf351803825400d6a25940919e70ae20f27dcfe7"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
