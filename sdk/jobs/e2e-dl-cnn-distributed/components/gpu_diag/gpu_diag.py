#import required libraries
from azure.ml import MLClient
from azure.ml.entities import Environment, CommandJob
from azure.ml.entities._assets.environment import BuildContext
from azure.identity import InteractiveBrowserCredential

# get a handle to the workspace
ml_client = MLClient(
    InteractiveBrowserCredential(), 
    subscription_id = '<SUBSCRIPTION_ID>', 
    resource_group = '<RESOURCE_GROUP>', 
    workspace = '<AML_WORKSPACE_NAME>'
)

# register environment based on dockerfile
env_docker_image = Environment(
    build=BuildContext(local_path="."),
    name="nccl-test-env",
    description="NCCL test enviroyment"
)
#ml_client.create_or_update(env_docker_image)

# create a job based on a simple command
job = CommandJob(
    # option 1: run mpirun yourself
    #command = "mpirun --allow-run-as-root --np ${{inputs.processes}} --bind-to numa --map-by ppr:${{inputs.processes}}:node /nccl-tests/build/all_reduce_perf -b %{{inputs.minbytes}} -e ${{inputs.maxbytes}} -f ${{inputs.stepfactor}} -g ${{inputs.ngpus}}",

    # option 2: rely on AzureML distribution
    command = "nvcc --version ; nvidia-smi ; /nccl-tests/build/all_reduce_perf -e ${{inputs.maxbytes}} -f ${{inputs.stepfactor}} -g ${{inputs.ngpus}}",
    distribution = {
        "distribution_type" : "Mpi",
        "process_count_per_instance" : 2
    },

    inputs = {
        'ngpus': 1,
        'minbytes': 8,
        'maxbytes': "8G",
        'stepfactor': 2,
        'processes': 2
    },
    environment = env_docker_image, #"nccl-test-env:2",
    compute = 'nv24-m60',
    display_name = 'gpu_diag',
    description = 'GPU Diag using NCCL tests',
    environment_variables = {
        'AZUREML_COMPUTE_USE_COMMON_RUNTIME': "true",
        'NCCL_DEBUG': "INFO",
        'NCCL_IB_PCI_RELAXED_ORDERING': "1",
        'UCX_TLS': "tcp",
        'UCX_NET_DEVICES': "eth0",
        'CUDA_DEVICE_ORDER': "PCI_BUS_ID",
        'NCCL_SOCKET_IFNAME': "eth0",
        'NCCL_TOPO_FILE': "/opt/microsoft/ndv4-topo.xml",
    }
)

# submit the command job
returned_job = ml_client.create_or_update(job)

# get a URL for the status of the job
print("*** To get to job: {}".format(returned_job.services["Studio"].endpoint))
