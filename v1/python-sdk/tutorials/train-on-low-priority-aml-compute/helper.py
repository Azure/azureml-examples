from azureml.core import Environment, ScriptRunConfig
from azureml.core.runconfig import PyTorchConfiguration
from azureml.data import OutputFileDatasetConfig


def launch_run(
    experiment, compute_target, num_epochs=1, output_dataset_storage_path=None
):
    """Launch a run training MNIST on remote compute."""
    ws = experiment.workspace
    dstore = ws.get_default_datastore()

    env = Environment.get(ws, "AzureML-ACPT-pytorch-1.11-py38-cuda11.3-gpu")
    distributed_config = PyTorchConfiguration(process_count=1)

    # Set output dataset used for model checkpointing for low-priority runs
    output_dataset_destination = None
    if output_dataset_storage_path:
        output_dataset_destination = (dstore, output_dataset_storage_path)
    output_dataset_config = OutputFileDatasetConfig(
        name="model_checkpoints",
        destination=output_dataset_destination,
        source="model_checkpoints/",
    )

    src = ScriptRunConfig(
        source_directory="./training_script",
        script="training_script.py",
        arguments=[output_dataset_config, "--num_epochs", num_epochs],
        compute_target=compute_target,
        environment=env,
        distributed_job_config=distributed_config,
    )

    run = experiment.submit(src)
    return run
