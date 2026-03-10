from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import json

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except:
    credential = InteractiveBrowserCredential()

# Use explicit credentials instead of config file
SUBSCRIPTION = "b17253fa-f327-42d6-9686-f3e553e24763"
RESOURCE_GROUP = "amlsdkv2022603"
WS_NAME = "amlsdkv2022603-ws"

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WS_NAME,
)

# List recent jobs
jobs = list(ml_client.jobs.list(max_results=10))
print(f"Found {len(jobs)} recent jobs\n")

for job in jobs:
    exp_name = getattr(job, 'experiment_name', '')
    if 'pipeline_samples' in str(exp_name):
        print(f"Found pipeline job: {job.name}")
        print(f"Status: {job.status}")
        print(f"Type: {job.type}")
        print()
        
        # Get the job details
        full_job = ml_client.jobs.get(job.name)
        
        # Print job details
        if hasattr(full_job, 'jobs'):
            print("Child jobs:")
            for step_name, step_details in full_job.jobs.items():
                status = getattr(step_details, 'status', 'unknown')
                print(f"  {step_name}: {status}")
                
                if 'score' in step_name.lower() and status != 'Completed':
                    print(f"\n    Getting logs for {step_name}...")
                    try:
                        # Try to get the output
                        print(f"    Job ID: {getattr(step_details, 'id', 'N/A')}")
                    except Exception as e:
                        print(f"    Error: {e}")
        break
else:
    print("No pipeline_samples jobs found")
