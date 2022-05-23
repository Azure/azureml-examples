import os
from pathlib import Path
from mldesigner import command_component, Input, Output


@command_component(
    name="train_image_classification_keras",
    version="1",
    display_name="Train Image Classification Keras",
    description="train image classification with keras",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    ),
)
def keras_train_component(
    input_data: Input(type="uri_folder"),
    output_model: Output(type="uri_folder"),
    epochs=10,
):
    # avoid dependency issue, execution logic is in train() func in train.py file
    from train import train

    train(input_data, output_model, epochs)
