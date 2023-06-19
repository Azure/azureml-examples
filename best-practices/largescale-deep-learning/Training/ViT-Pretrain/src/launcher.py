# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
Sample entry script for large scale training of Computer Vision models on Azure.

When running the script for the first time, set the BUILD_ENVIRONMENT variable
below to True to create an Azure custom environment. Then set it to False when
launching actual training runs.
"""

from azure.ai.ml import MLClient, Input, command, PyTorchDistribution
from azure.ai.ml.entities import Environment

from azure.identity import DefaultAzureCredential


BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest"
CONDA_FILE_NAME = "./conda.yml"
CODE_DIRECTORY_NAME = "./"
COMMAND = (
    "python run_image_classification.py --dataset_directory_name ${{inputs.data}} "
    "--num_epochs ${{inputs.num_epochs}} --batch_size ${{inputs.batch_size}} --num_workers ${{inputs.num_workers}} "
    "--num_nodes ${{inputs.num_nodes}} --num_devices ${{inputs.num_devices}} --strategy ${{inputs.strategy}}"
)

# Enter the details of your Azure ML workspace.
SUBSCRIPTION_ID = "<SUBSCRIPTION_ID>"
RESOURCE_GROUP = "<RESOURCE_GROUP_NAME>"
WORKSPACE_NAME = "<WORKSPACE_NAME>"
CLUSTER_NAME = "<CLUSTER_NAME>"
ENVIRONMENT_NAME = "<ENVIRONMENT_NAME>"
EXPERIMENT_NAME = "<EXPERIMENT_NAME>"

NUM_NODES = 8
NUM_GPUS_PER_NODE = 4
SHARED_MEMORY_SIZE_STR = "100G"
DATA_MODE = "mount_no_caching"  # alternative: "download"

# Run the script with BUILD_ENVIRONMENT=True first, then set BUILD_ENVIRONMENT=False for subsequent runs.
# BUILD_ENVIRONMENT = True
BUILD_ENVIRONMENT = False

DATASET = "imagenet"

NUM_EPOCHS = 10
BATCH_SIZE = 92
NUM_WORKERS = 6


if __name__ == "__main__":
    # Create the MLClient object.
    credential = DefaultAzureCredential(
        exclude_shared_token_cache_credential=True,
        exclude_visual_studio_code_credential=False,
    )
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    if BUILD_ENVIRONMENT:
        # Build the environment first.
        environment = Environment(
            name=ENVIRONMENT_NAME,
            description="Custom environment for large scale training of Computer Vision models.",
            tags={},
            conda_file=CONDA_FILE_NAME,
            image=BASE_IMAGE,
        )
        environment = ml_client.environments.create_or_update(environment)

    else:
        # After the environment is built, submit a run that trains an image classification model.

        # Set job parameters.
        command_parameters = dict(
            code=CODE_DIRECTORY_NAME,
            command=COMMAND,
            inputs=dict(
                data=Input(
                    type="uri_folder",
                    path="azureml://datastores/datasets/paths/" + DATASET,
                    mode="ro_mount" if DATA_MODE.startswith("mount") else "download",
                ),
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                num_nodes=NUM_NODES,
                num_devices=NUM_GPUS_PER_NODE,
                strategy="ddp",
            ),
            environment=ENVIRONMENT_NAME + "@latest",
            shm_size=SHARED_MEMORY_SIZE_STR,
            compute=CLUSTER_NAME,
            instance_count=NUM_NODES,
            experiment_name=EXPERIMENT_NAME,
        )
        if DATA_MODE == "mount_no_caching":
            command_parameters.update(
                dict(
                    environment_variables=dict(
                        DATASET_MOUNT_BLOCK_BASED_CACHE_ENABLED=True,  # enable block-based caching
                        DATASET_MOUNT_BLOCK_FILE_CACHE_ENABLED=False,  # disable caching on disk
                        DATASET_MOUNT_MEMORY_CACHE_SIZE=0,  # disabling in-memory caching
                    )
                )
            )
        if NUM_NODES > 1:
            command_parameters.update(
                dict(
                    distribution=PyTorchDistribution(
                        node_count=NUM_NODES,
                        process_count_per_instance=NUM_GPUS_PER_NODE,
                    )
                )
            )

        # Submit the job.
        job = command(**command_parameters)
        ml_client.create_or_update(job)
