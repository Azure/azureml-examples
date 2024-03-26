## Run Node Health Checks (NHC)
This command job is used to test compute nodes being used for LLM training. During training, issues with the compute nodes can cause problems that may affect the training in unexpected ways. Oftentimes it can be hard to determine if the compute nodes are the source of the problem. To pinpoint if compute nodes are failing or problematic, these node health checks can be run.

### When to use NHC
- Mismatched outputs between nodes in output files. If the output files for nodes show that different nodes are on different training steps, this could indicate problems with communication between nodes.
- AzureML GPU metrics show low GPU utilization
- Data loading seems abnormally slow
- Out-of-Memory errors in logs

### Expected Output
The job should typically complete quickly, likely under 5 minutes. Results of the job can be seen in the user logs folder under the "outputs + logs" tab of the job page. At the bottom of the ```std_log_process_0.txt``` file, you will see a message that says 'Health Checks completed' once the checks are done. A list of the checks run will also be shown at the bottom of the log file with a short description of each check. If one or more of the checks fails, the job will fail and the failing checks will be shown in the job overview tab like below: 

![image](https://github.com/Azure/azureml-examples/assets/73311224/e4d8df01-40ff-4b3f-8560-8182dd427287)

When a job fails, messages will be shown in the logs that will look similar to this:
![image](https://github.com/Azure/azureml-examples/assets/73311224/aa31c29c-9669-40e4-acb1-52d7843f8a56)

The job can be run on multiple nodes, so there will be a log file for each node. Some nodes may pass all the checks without fail and some nodes will fail the tests. If a node fails any of the checks, it will also be automatically kicked from the cluster and a good node will be reallocated. The job will succeed without errors if all health checks have passed on all nodes.

### How To Run
> This readme assumes azureml-cli is installed with the az-ai-ml extension. It also assumes you are logged into a workspace and subscription.
1. Create a compute in the AzureML Studio with the type of nodes you want to run a health check on. In the health_job.yaml file, specify the following arguments:
    - The "compute" argument with the name of the compute you created.
    - The "instance_count" resource variable with the number of compute nodes you want to run the job on.
3. Start the command job by running the command: ```az ml job create --file health_job.yaml```