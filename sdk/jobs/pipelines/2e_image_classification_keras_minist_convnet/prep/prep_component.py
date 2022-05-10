from pathlib import Path
from mldesigner import command_component, Input, Output


@command_component(
    name="prep_data",
    version="1",
    display_name="Prep Data",
    description="Convert data to CSV file, and split to training and test data",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    ),
)
def prep(
    input_data: Input,
    training_data: Output,
    test_data: Output,
):
    # Avoid dependency issue, execution logic is in prep.py file
    from prep import prep

    prep(input_data, training_data, test_data)
