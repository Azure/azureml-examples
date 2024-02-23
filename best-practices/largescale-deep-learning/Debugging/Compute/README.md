## Run Node Health Checks
> This readme assumes azureml-cli is installed with the az-ai-ml extension. It also assumes you are logged into a workspace and subscription.
1. Create the environment to be used by the command job using the command ```az ml environment create --file env.yaml```
2. Create a compute in the AzureML Studio with the type of nodes you want to run a health check on. In the health_job.yaml file, specify the following arguments:
    - The "compute" argument with the name of the compute you created.
    - The "instance_count" resource variable with the number of compute nodes you want to run the job on.
3. Start the command job by running the command: ```az ml job create --file health_job.yaml```