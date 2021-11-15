from azure.core.exceptions import ResourceNotFoundError
from azure.ml import dsl, MLClient
from azure.ml.dsl import Pipeline
from azure.ml.entities import Dataset
from azure.ml.entities import JobInputDataset
from azure.ml.entities import InputDatasetEntry
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline(client: MLClient) -> Pipeline:
    # 1. Load component funcs
    basic_func = dsl.load_component(yaml_file=parent_dir + "./component.yml")

    # 2. Construct pipeline
    @dsl.pipeline(
        description="Example of using data from a Dataset as pipeline input",
    )
    def sample_pipeline(
            pipeline_sample_input_data,
            pipeline_sample_input_string,
    ):
        hello_python_world_job = basic_func(
            sample_input_data=pipeline_sample_input_data,
            sample_input_string=pipeline_sample_input_string,
        )
        hello_python_world_job.compute = "cpu-cluster"
        return {
            "pipeline_sample_output_data": hello_python_world_job.outputs.sample_output_data,
        }
    try:
        dataset = client.datasets.get("sampledata1234", "1")
    except ResourceNotFoundError:
        # Create the data version if not exits
        data = Dataset.load(path=parent_dir + "./data.yml")
        dataset = client.datasets.create_or_update(data)
    pipeline = sample_pipeline(InputDatasetEntry(dataset=dataset), "Hello_Pipeline_World")
    pipeline.outputs.pipeline_sample_output_data.mode = "upload"
    return pipeline
