from azure.ml import dsl
from azure.ml.dsl import Pipeline
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline() -> Pipeline:
    # 1. Load component funcs
    tf_func = dsl.load_component(yaml_file=parent_dir + "./component.yml")

    # 2. Construct pipeline
    @dsl.pipeline(
        description="Train using TF component",
    )
    def sample_pipeline():
        tf_job = tf_func(epochs=1)
        tf_job.compute = "cpu-cluster"
        tf_job.outputs.trained_model_output.mode = "upload"
        tf_job.distribution.worker_count = 1
        tf_job.distribution.distributionType = "tensorflow"
        tf_job.resources.instance_count = 2

    pipeline = sample_pipeline()
    return pipeline
