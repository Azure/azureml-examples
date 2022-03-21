import os, json
from pathlib import Path

from src.train import mnist_dataset, get_worker_model, write_filepath
from azure.ml import dsl
from azure.ml.entities import Environment
from azure.ml.dsl._types import DataInput, DataOutput

conda_env = Environment(
    conda_file=Path(__file__).parent / "conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)
@dsl.command_component(
    name="TF_mnist",
    version="1",
    display_name="TF_mnist",
    description="Train a basic neural network with TensorFlow on the MNIST dataset, distributed via TensorFlow.",
    environment=conda_env,
    distribution={
        "type": "tensorflow",
        "worker_count": 2
    },
    code=".."
)
def tf_func(
    trained_model_output: DataOutput, 
    epochs=3,
    steps_per_epoch=70,
    per_worker_batch_size=64,
):
    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])

    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)
    multi_worker_model = get_worker_model()
    multi_worker_model.fit(
        multi_worker_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    
    # Save the model
    task_type, task_id = (tf_config["task"]["type"], tf_config["task"]["index"])
    write_model_path = write_filepath(trained_model_output, task_type, task_id)
    multi_worker_model.save(write_model_path)
