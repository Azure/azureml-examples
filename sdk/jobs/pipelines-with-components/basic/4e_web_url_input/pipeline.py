from azure.ml import dsl
from azure.ml.dsl import Pipeline
from azure.ml.entities import JobInputUri
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline() -> Pipeline:
    # 1. Load component funcs
    basic_func = dsl.load_component(yaml_file=parent_dir + "./component.yml")

    # 2. Construct pipeline
    @dsl.pipeline(
        description="Example of using a file hosted at a web URL as pipeline input",
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

    pipeline = sample_pipeline(
        JobInputUri(file="https://dprepdata.blob.core.windows.net/demo/Titanic.csv"),
        "Hello_Pipeline_World",
    )
    return pipeline
