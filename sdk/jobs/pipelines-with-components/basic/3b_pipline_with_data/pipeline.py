from azure.ml import dsl
from azure.ml.dsl import Pipeline
from azure.ml.entities import Dataset
from pathlib import Path

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline() -> Pipeline:
    # 1. Load component funcs
    a_func = dsl.load_component(yaml_file=parent_dir + "./componentA.yml")
    b_func = dsl.load_component(yaml_file=parent_dir + "./componentB.yml")
    c_func = dsl.load_component(yaml_file=parent_dir + "./componentC.yml")

    # 2. Construct pipeline
    @dsl.pipeline(compute="cpu-cluster")
    def sample_pipeline(pipeline_sample_input_data):
        componentA_job = a_func(componentA_input=pipeline_sample_input_data)
        componentB_job = b_func(componentB_input=componentA_job.outputs.componentA_output)
        componentC_job = c_func(componentC_input=componentB_job.outputs.componentB_output)
        return {
            "pipeline_sample_output_data_A": componentA_job.outputs.componentA_output,
            "pipeline_sample_output_data_B": componentB_job.outputs.componentB_output,
            "pipeline_sample_output_data_C": componentC_job.outputs.componentC_output,
        }

    pipeline = sample_pipeline(Dataset(local_path=parent_dir + "./data/"))
    pipeline.outputs.pipeline_sample_output_data_A.mode = "upload"
    pipeline.outputs.pipeline_sample_output_data_B.mode = "rw_mount"
    pipeline.outputs.pipeline_sample_output_data_C.mode = "upload"
    return pipeline
