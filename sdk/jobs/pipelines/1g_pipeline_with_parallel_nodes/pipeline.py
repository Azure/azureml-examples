from azure.ml import dsl, Input
from azure.ml.constants import AssetTypes, InputOutputModes
from azure.ml.entities import PipelineJob
from azure.ml.entities import Component as ComponentEntity
from pathlib import Path
from typing import Callable

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline(randstr: Callable[[], str]) -> PipelineJob:
    # 1. Load component funcs
    get_data = ComponentEntity.load(
        path=parent_dir + "/get_data.yml"
    )

    file_batch_inference1 = ComponentEntity.load(
        path=parent_dir + "/score.yml", params_override=[{"name": randstr()}]
    )
    file_batch_inference2 = ComponentEntity.load(
        path=parent_dir + "/score.yml", params_override=[{"name": randstr()}]
    )
    tabular_batch_inference1 = ComponentEntity.load(
        path=parent_dir + "/tabular_input_e2e.yml"
    )

    # Construct pipeline
    @dsl.pipeline(default_compute="cpu-cluster")
    def parallel_in_pipeline(pipeline_job_data_path, pipeline_score_model):
        get_data_node = get_data(input_data=pipeline_job_data_path)
        get_data_node.outputs.file_output_data.type = AssetTypes.MLTABLE
        get_data_node.outputs.tabular_output_data.type = AssetTypes.MLTABLE

        batch_inference_node1 = file_batch_inference1(job_data_path=get_data_node.outputs.file_output_data)
        batch_inference_node1.inputs.job_data_path.mode = InputOutputModes.EVAL_MOUNT
        batch_inference_node1.outputs.job_output_path.type = AssetTypes.MLTABLE

        batch_inference_node2 = file_batch_inference2(job_data_path=batch_inference_node1.outputs.job_output_path)
        batch_inference_node2.inputs.job_data_path.mode = InputOutputModes.EVAL_MOUNT

        tabular_batch_inference_node1 = tabular_batch_inference1(
            job_data_path=get_data_node.outputs.tabular_output_data,
            score_model=pipeline_score_model
        )
        tabular_batch_inference_node1.inputs.job_data_path.mode = InputOutputModes.DIRECT

        return {
            "pipeline_job_out_path_1": batch_inference_node2.outputs.job_output_path,
            "pipeline_job_out_path_2": tabular_batch_inference_node1.outputs.job_out_path,
        }

    pipeline = parallel_in_pipeline(
        pipeline_job_data_path=Input(
            type=AssetTypes.MLTABLE, path=parent_dir+"/dataset/", mode=InputOutputModes.RW_MOUNT
        ),
        pipeline_score_model=Input(
            path=parent_dir+"/model/", type=AssetTypes.URI_FOLDER, mode=InputOutputModes.DOWNLOAD
        ),
    )

    return pipeline
