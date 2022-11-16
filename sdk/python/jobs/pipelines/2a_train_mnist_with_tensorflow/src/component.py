import os, json
from pathlib import Path


from mldesigner import command_component, Output


@command_component(
    name="tf_mnist",
    version="1",
    display_name="TF_mnist",
    description="Train a basic neural network with TensorFlow on the MNIST dataset, distributed via TensorFlow.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.0.1-ubuntu20.04",
    ),
    distribution={"type": "tensorflow", "worker_count": 2},
)
def train_tf(
    trained_model_output: Output(type="uri_folder"),
    epochs=3,
    steps_per_epoch=70,
    per_worker_batch_size=64,
):
    # avoid dependency issue
    from train import train_and_save_model

    train_and_save_model(
        per_worker_batch_size, epochs, steps_per_epoch, trained_model_output
    )
