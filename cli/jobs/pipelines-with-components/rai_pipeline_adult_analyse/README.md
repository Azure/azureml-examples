# Running a RAI sample pipeline
In this example, we will explain how to use Responsible API in a pipeline. In order to run this pipeline, please run these commands to register RAI data and environment: 
- `az ml environment create --file environment/responsibleai-environment.yaml` 
- `az ml data create --file data/data_adult_train.yaml` 
- `az ml data create --file data/data_adult_test.yaml` 

Learn more on [Responsible API](https://github.com/Azure/RAI-vNext-Preview).
