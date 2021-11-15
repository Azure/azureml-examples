from azure.ml import dsl
from azure.ml.dsl import Pipeline
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline() -> Pipeline:
    # 1. Load component funcs
    pytorch_func = dsl.load_component(
        yaml_file=parent_dir + "./component.yml"
    )

    # 2. Construct pipeline
    @dsl.pipeline(
        description="Prints the environment variables useful for scripts "
                    "running in a PyTorch training environment",
    )
    def sample_pipeline():
        pytorch_job = pytorch_func()
        pytorch_job.compute = "cpu-cluster"
        pytorch_job.distribution.process_count_per_instance = 1
        pytorch_job.distribution.distributionType = "pytorch"
        pytorch_job.resources.instance_count = 2

    pipeline = sample_pipeline()
    return pipeline
