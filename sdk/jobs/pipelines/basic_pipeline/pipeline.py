from azure.ml import dsl
from azure.ml.dsl import Pipeline
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline() -> Pipeline:
    # 1. Load component funcs
    a_func = dsl.load_component(yaml_file=parent_dir + "./componentA.yml")
    b_func = dsl.load_component(yaml_file=parent_dir + "./componentB.yml")
    c_func = dsl.load_component(yaml_file=parent_dir + "./componentC.yml")

    # 2. Construct pipeline
    @dsl.pipeline(
        compute="cpu-cluster",
        description="Basic Pipeline Job with 3 Hello World components",
    )
    def sample_pipeline():
        componentA_job = a_func()
        componentB_job = b_func()
        componentC_job = c_func()

    pipeline = sample_pipeline()
    return pipeline
