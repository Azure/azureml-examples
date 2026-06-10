# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""
Install necessary libraries and packages required for running AML training jobs.
"""
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

# Import necessary classes for environment management
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Data,
    AmlCompute,
    Environment,
    BuildContext,
)
from azure.core.exceptions import ResourceNotFoundError
from rich import print
from azure.identity import InteractiveBrowserCredential, DefaultAzureCredential

try:
    credential = DefaultAzureCredential()
except Exception:
    # Use interactive browser-based authentication
    credential = InteractiveBrowserCredential()

# Input your Azure subscription ID, resource group name, and workspace name.
# You can find these values in the Azure portal.
AML_SUBSCRIPTION = "<SUBSCRIPTION_ID>"
AML_RESOURCE_GROUP = "<RESOURCE_GROUP_NAME>"
AML_WORKSPACE_NAME = "<WORKSPACE_NAME>"
N_NODES = 2  # Number of nodes to use for training, options: [1, 2]

# Initialize the MLClient to connect to your Azure ML workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=AML_SUBSCRIPTION,
    resource_group_name=AML_RESOURCE_GROUP,
    workspace_name=AML_WORKSPACE_NAME,
)


def setup_dataset():
    """
    Register the medical MCQA dataset in the Azure ML workspace.
    The datasets will be referred from the ./datasets folder.
    """
    med_mcqa_dataset = "dataset-med-mcqa"
    med_mcqa_data_path = "./datasets/med_mcqa"
    med_mcqa_data = ml_client.data.create_or_update(
        Data(
            path=med_mcqa_data_path,
            type=AssetTypes.URI_FOLDER,
            description="Training dataset with medical mcqa data",
            name=med_mcqa_dataset,
        )
    )
    print(f"✅ Dataset {med_mcqa_dataset} created in AML workspace.")
    return med_mcqa_data


def setup_model():
    """
    Download and Register the model in the Azure ML workspace.
    """
    # Model details
    model_id = "Qwen/Qwen2.5-7B-Instruct"  # Hugging Face model ID
    download_path = "./models"  # Local download path
    model_name = "Qwen2_5-7B-Instruct_base"  # Name for Azure ML registration

    # Download model from Hugging Face and register in Azure ML workspace
    try:
        model = ml_client.models.get(name=model_name, label="latest")
        print(f"✅ Selected model {model_name} from AML workspace.")
    except ResourceNotFoundError:
        print(f"❌ Model {model_name} not found, downloading and registering...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            local_dir=download_path,
            # token=os.environ["HF_TOKEN"]  # Uncomment if authentication is needed
        )
        model = ml_client.models.create_or_update(
            Model(name=model_name, path=download_path, type=AssetTypes.CUSTOM_MODEL)
        )
        print(f"✅ Model registered in AML workspace: {model.id}")
    return model


def setup_compute():
    """
    Create a compute cluster in the Azure ML workspace.
    """
    # Compute Cluster Setup: Select or Create GPU Compute for Training

    # Specify the desired Azure VM size (default: 8 x H100 GPUs). This job requires flash attention and needs A100 or H100 GPUs.
    compute_cluster_size = "STANDARD_ND96ISR_H100_V5"

    # Name of the compute cluster to use (change if you have a different cluster)
    compute_cluster = "grpo-h100-2"

    try:
        # Try to get the existing compute cluster
        compute = ml_client.compute.get(compute_cluster)
        print(f"✅ Selected compute cluster {compute_cluster} from AML workspace.")
    except Exception:
        print(
            f"❌ Compute cluster '{compute_cluster}' not found. Creating a new one ({compute_cluster_size})..."
        )
        try:
            print("🔄 Creating dedicated GPU compute cluster...")
            compute = AmlCompute(
                name=compute_cluster,
                size=compute_cluster_size,
                tier="Dedicated",
                max_instances=N_NODES,
                min_instances=N_NODES,
            )
            ml_client.compute.begin_create_or_update(compute).wait()
            print("✅ Compute cluster created successfully.")
        except Exception as e:
            print(f"🚨 Error: {e}")
            raise ValueError(
                f"WARNING! Compute size '{compute_cluster_size}' not available in this workspace."
            )

    # Final check: Ensure the compute cluster is provisioned and ready
    compute = ml_client.compute.get(compute_cluster)
    if compute.provisioning_state.lower() == "failed":
        raise ValueError(
            f"🚫 Provisioning failed: Compute '{compute_cluster}' is in a failed state. "
            f"Please try creating a different compute cluster."
        )
    return compute


def setup_environment():
    """
    Setup environment for the training job.
    """
    # Define the environment name
    env_name = "grpo-training-env"

    try:
        # Try to get the existing environment from Azure ML workspace
        environment = ml_client.environments.get(name=env_name, label="latest")
        print(f"✅ Selected environment {env_name} from AML workspace.")
    except ResourceNotFoundError as e:
        # If not found, create a new environment using the specified build context
        print(f"❌ Environment {env_name} not found, creating a new one.")
        environment = ml_client.environments.create_or_update(
            Environment(
                build=BuildContext(path="./environment"),
                name=env_name,
                description="Environment for grpo trainer.",
            )
        )
        print(f"✅ Environment registered in AML workspace: {environment.id}")
    return environment


def setup():
    """
    Main function to setup the Azure ML workspace with dataset, model, compute, and environment.
    """
    # Register the dataset
    med_mcqa_data = setup_dataset()

    # Setup the model
    model = setup_model()

    # Setup the compute cluster
    compute = setup_compute()

    # Setup the environment
    environment = setup_environment()

    print("✅ Setup completed successfully !!!")
    return ml_client, med_mcqa_data, model, compute, environment
