# Online Endpoints Model Profiler

To run this example end-to-end, execute the script [`deploy-moe-profiler.sh`](../../../../deploy-moe-profiler.sh). 

## Overview

Inferencing machine learning models is a time and compute intensive process. It is vital to quantify the performance of model inferencing to ensure that you make the best use of compute resources and reduce cost to reach the desired performance SLA (e.g. latency, throughput).

Online Endpoints Model Profiler (Preview) provides fully managed experience that makes it easy to benchmark your model performance served through [Online Endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/concept-endpoints).

* Use the benchmarking tool of your choice.

* Easy to use CLI experience.
  
* Support for CI/CD MLOps pipelines to automate profiling.
  
* Thorough performance report containing latency percentiles and resource utilization metrics.

## A brief introduction on benchmarking tools

The online endpoints model profiler currently supports 3 types of benchmarking tools: wrk, wrk2, and labench.

* `wrk`: wrk is a modern HTTP benchmarking tool capable of generating significant load when run on a single multi-core CPU. It combines a multithreaded design with scalable event notification systems such as epoll and kqueue. For detailed info please refer to this link: https://github.com/wg/wrk.

* `wrk2`: wrk2 is wrk modifed to produce a constant throughput load, and accurate latency details to the high 9s (i.e. can produce accuracy 99.9999% if run long enough). In addition to wrk's arguments, wrk2 takes a throughput argument (in total requests per second) via either the --rate or -R parameters (default is 1000). For detailed info please refer to this link: https://github.com/giltene/wrk2.

* `labench`: LaBench (for LAtency BENCHmark) is a tool that measures latency percentiles of HTTP GET or POST requests under very even and steady load. For detailed info please refer to this link: https://github.com/microsoft/LaBench.
  
## Prerequisites

* Azure subscription. If you don't have an Azure subscription, sign up to try the [free or paid version of Azure Machine Learning](https://azure.microsoft.com/free/) today.

* Azure CLI and ML extension. For more information, see [Install, set up, and use the CLI (v2) (preview)](how-to-configure-cli.md).

* A compute instance with Contributor access to run the profiler or sufficient user permissions to create one (i.e. Ownership of a resource group.)

## Set variables

```bash
RAND=`echo $RANDOM`
ENDPOINT_NAME=endpt-moe-$RAND 
PROFILER_COMPUTE_NAME=profiler
PROFILER_COMPUTE_SIZE="Standard_DS4_v2"
```

## Create an online endpoint

```bash
az ml online-endpoint create -n $ENDPOINT_NAME
```

## Create an online deployment

* Replace the `instance_type` in deployment yaml file with your desired Azure VM SKU. VM SKUs vary in terms of computing power, price and availability in different Azure regions.

* Tune `request_settings.max_concurrent_requests_per_instance` which defines the concurrent level. The higher this setting is, the higher throughput the endpoint gets. If this setting is set higher than the online endpoint can handle, the inference request may end up waiting in the queue and eventually results in longer end-to-end latency.

* If you plan to profile using multiple `instance_type` and `request_settings.max_concurrent_requests_per_instance`, please create one online deployment for each pair. You can attach all online deployments under the same online endpoint.

Below is the [`deployment.yml`](deployment.yml) yaml file that defines an online deployment for this example.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: blue
endpoint_name: <ENDPOINT_NAME>
model:
  path: ../../model-1/model/
code_configuration: 
  code: ../../model-1/onlinescoring/
  scoring_script: score.py
environment:
  conda_file: ../../model-1/environment/conda.yml
  image: mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210727.v1
instance_type: Standard_DS2_v2
instance_count: 1
request_settings:
  request_timeout_ms: 3000
  max_concurrent_requests_per_instance: 1024
```

Create the deployment with the following command:

```bash
az ml online-deployment create -f endpoints/online/managed/profiler/deployment.yml \
    --set endpoint_name=$ENDPOINT_NAME \
    --all-traffic
``` 

## Create a compute to host the profiler

You will need a compute to host the profiler, send requests to the online endpoint and generate performance report.

* This compute is NOT the same one that you used above to deploy your model. Please choose a compute SKU with proper network bandwidth (considering the inference request payload size and profiling traffic, we'd recommend Standard_DS4_v2) in the same region as the online endpoint.

Below is the [`compute.yml`](compute.yml) file that defines a compute for this example:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/computeInstance.schema.json
name: <PROFILER_COMPUTE_NAME>
type: computeinstance
size: Standard_DS4_v2
identity:
  type: system_assigned
```

Create the compute with the following command: 

```bash
az ml compute create --name $PROFILER_COMPUTE_NAME --size $PROFILER_COMPUTE_SIZE --identity-type SystemAssigned --type amlcompute
```

### Create proper role assignment for accessing online endpoint resources.

The compute needs to have contributor role to the machine learning workspace. For more information, see [Assign Azure roles using Azure CLI](https://docs.microsoft.com/en-us/azure/role-based-access-control/role-assignments-cli).

  ```bash
  compute_info=`az ml compute show --name $PROFILER_COMPUTE_NAME --query '{"id": id, "identity_object_id": identity.principal_id}' -o json`
  workspace_resource_id=`echo $compute_info | jq -r '.id' | sed 's/\(.*\)\/computes\/.*/\1/'`
  identity_object_id=`echo $compute_info | jq -r '.identity_object_id'`
  az role assignment create --role Contributor --assignee-object-id $identity_object_id --scope $workspace_resource_id
  if [[ $? -ne 0 ]]; then echo "Failed to create role assignment for compute $PROFILER_COMPUTE_NAME" && exit 1; fi
  ```

## Understand a profiling job

A profiling job simulates how an online endpoint serves live requests. It produces a throughput load to the online endpoint and generates performance report.

Below is the [template yaml](profiling/job-template.yml) file that defines a profiling job. 

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: >
  python -m online_endpoints_model_profiler --payload_path ${{inputs.payload}}
experiment_name: profiling-job
display_name: <% SKU_CONNECTION_PAIR %>
environment:
  image: mcr.microsoft.com/azureml/online-endpoints-model-profiler:latest
environment_variables:
  ONLINE_ENDPOINT: "<% ENDPOINT_NAME %>"
  DEPLOYMENT: "<% DEPLOYMENT_NAME %>"
  PROFILING_TOOL: "<% PROFILING_TOOL %>"
  DURATION: "<% DURATION %>"
  CONNECTIONS: "<% CONNECTIONS %>"
  TARGET_RPS: "<% TARGET_RPS %>"
  CLIENTS: "<% CLIENTS %>"
  TIMEOUT: "<% TIMEOUT %>"
  THREAD: "<% THREAD %>"
compute: "azureml:<% COMPUTE_NAME %>"
inputs:
  payload:
    type: uri_file
    path: azureml://datastores/workspaceblobstore/paths/profiling_payloads/<% ENDPOINT_NAME %>_payload.txt
```

##### YAML syntax #####

| Key | Type  | Description | Allowed values | Default value |
| --- | ----- | ----------- | -------------- | ------------- |
| `command` | string | The command for running the profiling job. | `python -m online_endpoints_model_profiler ${{inputs.payload}}` | - |
| `experiment_name` | string | The experiment name of the profiling job. An experiment is a group of jobs. | - | - |
| `display_name` | string | The profiling job name. | - | A random string guid, such as `willing_needle_wrzk3lt7j5` |
| `environment.image` | string | An Azure Machine Learning curated image containing benchmarking tools and profiling scripts. | mcr.microsoft.com/azureml/online-endpoints-model-profiler:latest | - |
| `environment_variables` | string | Environment vairables for the profiling job. | [Profiling related environment variables](#YAML-profiling-related-environment_variables)<br><br>[Benchmarking tool related environment variables](#YAML-benchmarking-tool-related-environment_variables) | - |
| `compute` | string | The aml compute for running the profiling job. | - | - |
| `inputs.payload` | string | Payload file that is stored in an AML registered datastore. | [Example payload file content](https://github.com/Azure/azureml-examples/blob/xiyon/mir-profiling/cli/endpoints/online/profiling/payload.txt) | - |

##### YAML profiling related environment_variables #####    

<table>
<tr>
<td> Key </td> <td> Description </td> <td> Default Value </td>
</tr>
<tr>
<td> <code>SUBSCRIPTION</code> </td> <td> Used together with <code>RESOURCE_GROUP</code>, <code>WORKSPACE</code>, <code>ONLINE_ENDPOINT</code>, <code>DEPLOYMENT</code> to form the profiling target. </td> <td> Subscription of the profiling job </td>
</tr>
<tr>
<td> <code>RESOURCE_GROUP</code> </td> <td> Used together with <code>SUBSCRIPTION</code>, <code>WORKSPACE</code>, <code>ONLINE_ENDPOINT</code>, <code>DEPLOYMENT</code> to form the profiling target. </td> <td> Resource group of the profiling job </td>
</tr>
<tr>
<td> <code>WORKSPACE</code> </td> <td> Used together with <code>SUBSCRIPTION</code>, <code>RESOURCE_GROUP</code>, <code>ONLINE_ENDPOINT</code>, <code>DEPLOYMENT</code> to form the profiling target. </td> <td> AML workspace of the profiling job </td>
</tr>
<tr>
<td> <code>ONLINE_ENDPOINT</code> </td> 
<td> 
Used together with <code>SUBSCRIPTION</code>, <code>RESOURCE_GROUP</code>,  <code>WORKSPACE</code>, <code>DEPLOYMENT</code> to form the profiling target.<br>
<br>
If not provided, <code>SCORING_URI</code> will be used as the profiling target.<br>
<br>
If neither <code>OLINE_ENDPOINT</code>/<code>DEPLOYMENT</code> nor <code>SCORING_URI</code> is provided, an error will be thrown.
</td>
<td> - </td>
</tr>
<tr>
<td> <code>DEPLOYMENT</code> </td> 
<td> 
Used together with  <code>SUBSCRIPTION</code>, <code>RESOURCE_GROUP</code>,  <code>WORKSPACE</code>, <code>ONLINE_ENDPOINT</code> to form the profiling target.<br>
<br>
If not provided, <code>SCORING_URI</code> will be used as the profiling target.<br>
<br>
If neither <code>OLINE_ENDPOINT</code>/<code>DEPLOYMENT</code> nor <code>SCORING_URI</code> is provided, an error will be thrown. </td>
<td> - </td>
</tr>
<tr>
<td> <code>IDENTITY_ACCESS_TOKEN</code> </td>
<td> 
An optional aad token for retrieving online endpoint scoring_uri, access_key, and resource usage metrics. This will not be necessary for the following scenario:<br>
- The aml compute that is used to run the profiling job has contributor access to the workspace of the online endpoint.<br>
<br>
Users should keep in mind that it's recommended to assign appropriate permissions to the aml compute rather than providing this aad token, since the aad token might be expired during the process of the profiling job. 
</td>
<td> - </td>
</tr>
<tr>
<td> <code>SCORING_URI</code> </td> <td> Users are optional to provide this env var as instead of the <code>SUBSCRIPTION</code>/<code>RESOURCE_GROUP</code>/<code>WORKSPACE</code>/<code>ONLINE_ENDPOINT</code>/<code>DEPLOYMENT</code> combination to define the profiling target. Although, missing <code>ONLINE_ENDPOINT</code>/<code>DEPLOYMENT</code> info will lead to missing resource usage metrics in the final report. </td> <td> - </td>
</tr>
<tr>
<td> <code>SCORING_HEADERS</code> </td> <td> Users may use this env var to provide any special headers necessary when invoking the profiling target. </td>
<td> 

```json
{
    "Content-Type": "application/json",
    "Authorization": "Bearer ${ONLINE_ENDPOINT_ACCESS_KEY}",
    "azureml-model-deployment": "${DEPLOYMENT}"
}
```

</td>
</tr>
<tr>
<td> <code>PROFILING_TOOL</code> </td> <td> The name of the benchmarking tool. Currently support: <code>wrk</code>, <code>wrk2</code>, <code>labench</code> </td> <td> <code>wrk</code> </td>
</tr>
<tr>
<td> <code>PAYLOAD</code> </td> 
<td>
Users may use this param to provide a single string format payload data for invoking the profiling target. For example: <code>{"data": [[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]]}</code>.<br>
<br>
If <code>inputs.payload</code> is provided in the profiling job yaml file, this env var will be ignored.
</td>
<td> - </td>
</tr>
</table>

##### YAML benchmarking tool related environment_variables #####

| Key | Description | Default Value | wrk | wrk2 | labench |
| --- | ----------- | ------------- | --- | ---- | ------- |
| `DURATION` | Period of time for running the benchmarking tool. | `300s` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `CONNECTIONS` | No. of connections for the benchmarking tool. The default value will be set to the value of `max_concurrent_requests_per_instance` | `1` | :heavy_check_mark: | :heavy_check_mark: | :x: |
| `THREAD` | No. of threads allocated for the benchmarking tool. | `1` | :heavy_check_mark: | :heavy_check_mark: | :x: |
| `TARGET_RPS` | Target requests per second for the benchmarking tool. | `50` | :x: | :heavy_check_mark: | :heavy_check_mark: |
| `CLIENTS` | No. of clients for the benchmarking tool. The default value will be set to the value of `max_concurrent_requests_per_instance` | `1` | :x: | :x: | :heavy_check_mark: |
| `TIMEOUT` | Timeout in seconds for each request. | `10s` | :x: | :x: | :heavy_check_mark: |


## Create a profiling job

### Upload the payload

```bash
payload_path=$(az ml data create -f endpoints/online/managed/profiler/data-payload.yml \
                --query path -o tsv)
``` 

### Create the job

Update the profiling job yaml template with your own values and create a profiling job. This example uses the below partially-filled [`job-env.yml`](job-env.yml) as an example and fills in the rest via `--set`. 

```bash
job_name=$(az ml job create -f endpoints/online/managed/profiler/job-env.yml \
            --set display_name="$PROFILER_COMPUTE_SIZE:1" \
            --set compute="azureml:$PROFILER_COMPUTE_NAME" \
            --set environment_variables.ONLINE_ENDPOINT=$ENDPOINT_NAME \
            --set inputs.payload.path=$payload_path \
            --query name -o tsv)
```

### View the profiling job in AzureML Studio 

```bash
az ml job show -n $job_name --web 
``` 

### Stream job logs to the console

```bash
az ml job stream -n $job_name 
``` 

### Read the performance report

* Users may find profiling job info in the AML workspace studio, under "Experiments" tab.
  ![image](https://user-images.githubusercontent.com/14539980/163346104-034d225e-ab58-4018-b712-d247c32d8823.png)

* Users may also find job metrics within each individual job page, under "Metrics" tab.
  ![image](https://user-images.githubusercontent.com/14539980/163347463-d9508c45-d724-49fd-baae-97e099f0b4f6.png)

* Users may also find job report file within each individual job page, under "Outputs + logs" tab, file "outputs/report.json".
  ![image](https://user-images.githubusercontent.com/14539980/163347805-a0269135-f615-4a7b-a13c-35630f0cb77a.png)
  
* Users may also use the following cli to download all job output files.

  ```bash
  az ml job download --name $JOB_ID --download-path $JOB_LOCAL_PATH
  ```

## Cleanup

Delete the endpoint: 
```bash
az ml online-endpoint delete -n $ENDPOINT_NAME
``` 

Delete the compute:
```bash
az ml compute delete -n $PROFILER_COMPUTE_NAME 
```

## Contact us

For any questions, bugs and requests of new features, please contact us at [miroptprof@microsoft.com](mailto:miroptprof@microsoft.com)