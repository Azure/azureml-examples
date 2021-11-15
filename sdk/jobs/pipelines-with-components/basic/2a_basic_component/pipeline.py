from azure.ml import dsl
from azure.ml.dsl import Pipeline
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline() -> Pipeline:
    # 1. Load component funcs
    basic_func = dsl.load_component(
        yaml_file=parent_dir + "./component.yml"
    )

    # 2. Construct pipeline
    @dsl.pipeline(
        description="Hello World component example",
    )
    def sample_pipeline():
        hello_python_world_job = basic_func()
        hello_python_world_job.compute = "cpu-cluster"

    pipeline = sample_pipeline()
    return pipeline
