import os
from pathlib import Path
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.entities import Environment

conda_env = Environment(
    conda_file=Path(__file__).parent / "conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
)


@dsl.command_component(
    name="train_image_classification_keras",
    version="1",
    display_name="Train Image Classification Keras",
    description="train image classification with keras",
    environment=conda_env,
)
def keras_train_component(
    input_data: Input,
    output_model: Output,
    epochs=10,
):
    # avoid dependency issue, execution logic is in train() func in train.py file
    from train import train

    train(input_data, output_model, epochs)
