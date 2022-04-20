import os
from pathlib import Path
from azure.ml import dsl, ArtifactInput, ArtifactOutput
from azure.ml.entities import Environment

conda_env = Environment(
    conda_file=Path(__file__).parent / "conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)

@dsl.command_component(
    name="train_image_classification_keras",
    version="1",
    display_name="Train Image Classification Keras",
    description="train image classification with keras",
    environment=conda_env,
)
def keras_train(
    input_data: ArtifactInput,
    output_model: ArtifactOutput,
    epochs=10,
):
    # avoid dependency issue, execution logic is in prep.py file
    from train import train
    train(input_data, output_model, epochs)


