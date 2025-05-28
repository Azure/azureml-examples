### This directory hosts an example to train an Instruct model into a Reasoning model on AML using GRPO - a RFT technique

#### Repo structure:
- aml_setup.py: AzureML specific code relating to creation of dataset, model and environment. The workspace config is located here, make changes to this file before running the notebook.
- launch_grpo_command_job-med-mcqa-commented.ipynb: Entrypoint for the directory. In most cases, developers will just run this notebook; after adjusting the AzureML configuration.
- datasets/medmcqa: It has 3 jsonl files for train, test and validation. Each record in the jsonl has 2 important fields, **problem** (column) which is the prompt encouraging the model to do reasoning and the golden **solution**
- environment: This is the definition of the [AzureML environment](https://learn.microsoft.com/en-us/azure/machine-learning/concept-environments?view=azureml-api-2) in which the training job will run.
- src/
    - BldDemo_Reasoning_Train.py: Code relating to creating an instance of the GRPOTrainer class from trl with correct configurations.
    - grpo_trainer_callbacks.py: Code which converts the output models to MLflow model format. This conversation simplifies deployment as AzureML automatically generated inferencing environment and server for models in this format.
    - grpo_trainer_rewards.py: Rewards for the training, these are python functions which take an LLM response and grade it for accuracy and format.

Additionally, [this video](https://youtu.be/YOm_IQt3YWw?si=5nZzyy-PZyP9XFSU&t=1344) offers an overview based on the contents of this repository.