from azureml.core import Environment, ScriptRunConfig
from azureml.core.runconfig import PyTorchConfiguration
from azureml.data import OutputFileDatasetConfig


def launch_run(
    experiment,
    compute_target,
    num_epochs=1,
    output_dataset_storage_path=None):
        """Launch a run training MNIST on remote compute."""
        workspace = experiment.workspace
        datastore = workspace.get_default_datastore()

        env_gpu = Environment.get(workspace, 'AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu')
        distributed_config = PyTorchConfiguration(process_count=1)

        # Set output dataset used for model checkpointing for low-priority runs
        output_dataset_destination = None
        if output_dataset_storage_path:
            output_dataset_destination = (datastore, output_dataset_storage_path)
        output_dataset_config = OutputFileDatasetConfig('model_checkpoints', destination=output_dataset_destination, source='model_checkpoints/')

        config = ScriptRunConfig(source_directory='./training_script',
                                script='training_script.py',
                                arguments=[output_dataset_config, '--num_epochs', num_epochs],
                                compute_target=compute_target,
                                environment=env_gpu,
                                distributed_job_config=distributed_config)

        script_run = experiment.submit(config)
        return script_run
