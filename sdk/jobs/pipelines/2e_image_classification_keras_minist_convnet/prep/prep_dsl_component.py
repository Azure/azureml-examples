from pathlib import Path
from azure.ml import dsl, Input, Output
from azure.ml.entities import Environment

conda_env = Environment(
    conda_file=Path(__file__).parent / "conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
)


@dsl.command_component(
    name="prep_data",
    version="1",
    display_name="Prep Data",
    description="Convert data to CSV file, and split to training and test data",
    environment=conda_env,
)
def prep(
    input_data: Input,
    training_data: Output,
    test_data: Output,
):
    # Avoid dependency issue, execution logic is in prep.py file
    from prep import prep

    prep(input_data, training_data, test_data)
