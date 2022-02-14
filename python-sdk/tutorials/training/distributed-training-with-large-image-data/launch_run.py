from azureml.core import Dataset, Environment, ScriptRunConfig
from azureml.core.runconfig import PyTorchConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data import OutputFileDatasetConfig


def launch_run(
    experiment,
    cluster,
    node_count=1,
    download_images=False,
    subsample_first_n_images=None,
    num_epochs=1,
    output_dataset_storage_path=None):

        num_gpus = _vm_sizes_to_num_gpus[cluster.vm_size]

        workspace = experiment.workspace
        datastore = workspace.get_default_datastore()
        exp = experiment

        # env_gpu = Environment.get(workspace, 'my-openmpi4-pytorch', 6)
        env_gpu = Environment.from_dockerfile('distributed_training_pytorch_env', 'Dockerfile')
        distributed_config = PyTorchConfiguration(process_count=num_gpus * node_count, node_count=node_count)
        
        # Get MS COCO training set
        train_dataset = Dataset.get_by_name(workspace, 'coco_train')
        if subsample_first_n_images:
            train_dataset = train_dataset.take(subsample_first_n_images)
        train_datset_config = DatasetConsumptionConfig('coco_train', train_dataset)

        # Get MS COCO validation set
        valid_dataset = Dataset.get_by_name(workspace, 'coco_valid')
        valid_datset_config = DatasetConsumptionConfig('coco_valid', valid_dataset)

        # Set output dataset used for model checkpointing for low-priority runs
        output_dataset_destination = None
        if output_dataset_storage_path:
            output_dataset_destination = (datastore, output_dataset_storage_path)
        output_dataset_config = OutputFileDatasetConfig('model_checkpoints', destination=output_dataset_destination, source='model_checkpoints/')

        config = ScriptRunConfig(source_directory='./training_script',
                                script='training_script.py',
                                arguments=[
                                    train_datset_config,
                                    valid_datset_config,
                                    output_dataset_config,
                                    '--download_images',
                                    download_images,
                                    '--num_epochs',
                                    num_epochs],
                                compute_target=cluster,
                                environment=env_gpu,
                                distributed_job_config=distributed_config)

        script_run = exp.submit(config)
        return script_run


_vm_sizes_to_num_gpus = {
    'STANDARD_NC6': 1,
    'STANDARD_NC24': 4,
    'STANDARD_NC24R': 4,
    'STANDARD_ND6S': 1,
    'STANDARD_ND24S': 4,
    'STANDARD_ND24RS': 4,
    'STANDARD_NC6S_V2': 1,
    'STANDARD_NC24S_V2': 4,
    'STANDARD_NC24RS_V2': 4,
    'STANDARD_NC6S_V3': 1,
    'STANDARD_NC24S_V3': 4,
    'STANDARD_NC24RS_V3': 4
}
