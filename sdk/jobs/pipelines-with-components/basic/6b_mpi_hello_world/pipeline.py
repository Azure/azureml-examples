from azure.ml import dsl
from azure.ml.dsl import Pipeline
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline() -> Pipeline:
    # 1. Load component funcs
    mpi_func = dsl.load_component(yaml_file=parent_dir + "./component.yml")

    # 2. Construct pipeline
    @dsl.pipeline(description="Show the MPI training environment")
    def sample_pipeline():
        tf_job = mpi_func()
        tf_job.compute = "cpu-cluster"
        tf_job.distribution.process_count_per_instance = 1
        tf_job.distribution.distributionType = "mpi"
        tf_job.resources.instance_count = 2

    pipeline = sample_pipeline()
    return pipeline
