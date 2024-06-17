## Run Node Health Checks (NHC)
This command job is used to test for and remove unhealthy compute nodes in an AzureML Cluster. During training, issues with the compute nodes can cause problems that may affect the training in unexpected ways. Oftentimes it can be hard to determine if the compute nodes are the source of the problem. This command job will check for any unhealthy nodes in a cluster and optionally remove them from the cluster. 
For large-scale clusters, it is best to avoid scaling down as removing problematic nodes reduces the chance of issues. Instead, maintain minimum and maximum node counts and avoid enabling auto-scaling. This approach helps minimize recurring issues, particularly regarding IB performance problems in the backend infrastructure. Therefore, for large customers, it's advisable to run the job, remove problematic nodes, and avoid scaling down.

### What does it do?
A series of node heath checks will be run on each node in a cluster to check for any problems. If any nodes in the cluster fail a health check, the failing node will be kicked out of the cluster and a healthy node will be reallocated(Kicking the nodes out can be turned off and on with an environment variable, see 'How To Run' instructions). The exact health checks that are run depend on the type of compute node being used. The node health check descriptions and results will be outputed to the std_out file in the outputs after running the job. The full list of node health checks that may be used to test is the following:
#### Hardware Checks
- check_hw_cpuinfo: Compares the properties of the OS-detected CPU(s) to the expected values for the SKU type to ensure that the correct number of physical sockets, execution cores, and "threads" (or "virtual cores") are present and functioning on the system.
- check_hw_physmem: Compares the amount of physical memory (RAM) present in the system with the minimum and maximum expected values for the SKU type.
- check_hw_swap: Compares the total system virtual memory (swap) size with the minimum and maximum expected values for the SKU type.
- check_hw_ib: Determines whether or not an active Infiniband link is present with the expected data rate. Also checks that the Infiniband device type is correct and that the kernel drivers and userspace libraries are the same OFED version.
- check_hw_eth: Verifies that a particular Ethernet device is available.
- check_hw_topology: Checks that the hardware topology matches the expected topology for the SKU type.
#### GPU Checks
- check_gpu_count: Checks that the GPU count detected by nvidia-smi is equal to the expected GPU count of this SKU type.
- check_gpu_xid: Checks for GPU xid errors in the kernel log. These errors can occur if the driver is programming the GPU incorrectly or there is a corruption of the commands sent to the GPU.
- check_nvsmi_healthmon: Runs the nvidia healthmon test. nvidia-healthmon focuses on checking for software and system configuration issues with GPUs.
- check_gpu_bw: This check tests GPU bandwidth using NVBandwidth and compares the results to the expected bandwidth for the SKU type.
- check_gpu_ecc: Checks for GPU Error Correcting Code(ECC) errors. These indicate issues with GPU memory.
- check_gpu_clock_throttling: Checks the GPU clock throttling reasons for unexpected behavior.
- check_nccl_allreduce: Runs the NCCL allreduce test to test that the speed of inter-GPU communication is equal to the expected speed. This test failing can indicate issues with NVLink.
- check_nvlink_status: Checks that NVLink is enabled.
#### Performance Check
- check_cpu_stream: Check that the CPU memory bandwidth matches the expected value for the SKU type.
#### Additional IB Checks
- check_ib_bw_non_gdr: Checks that the Infiniband bandwidth matches the expected value for the SKU type.
- check_ib_bw_gdr: Checks that the Infiniband bandwidth matches the expected value for the SKU type.
- check_nccl_allreduce_ib_loopback: Checks for Infiniband issues by running NCCL allreduce and disabling NCCL shared memory.
- check_ib_link_flapping: Checks for Infiniband link flapping within a specified time interval. None should occur.

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
    - The "KICK_BAD_NODE" environment variable with 'true' or 'false' depending on whether or not you want the job to remove an unhealthy node from your cluster if one is discovered and replace it with a healthy node. (Warning: removing a node from a cluster and replacing it can take several minutes. Sometimes deallocating and reallocating the cluster manually is faster.)
3. Start the command job by running the command: ```az ml job create --file health_job.yaml```