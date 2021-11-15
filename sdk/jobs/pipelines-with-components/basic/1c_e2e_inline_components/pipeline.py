from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os

from azure.ml import dsl
from azure.ml.dsl import Pipeline
from azure.ml.entities import Dataset, Code, CommandComponent

parent_dir = str(Path(__file__).parent)


def generate_dsl_pipeline() -> Pipeline:
    environment = "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:5"

    # 1. Construct pipeline
    @dsl.pipeline(
        compute="cpu-cluster",
        description="E2E dummy train-score-eval pipeline with components defined inline in pipeline job",
    )
    def sample_pipeline(
            pipeline_job_training_input,
            pipeline_job_test_input,
            pipeline_job_training_max_epocs,
            pipeline_job_training_learning_rate,
            pipeline_job_learning_rate_schedule,
    ):
        train_component = CommandComponent(
            name="Train_Inline",
            version="10",
            inputs=dict(
                training_data=dict(type="path"),
                max_epocs=dict(type="integer"),
                learning_rate=dict(type="number", default=0.01),
                learning_rate_schedule=dict(type="string", default="time-based")
            ),
            outputs=dict(
                model_output=dict(type="path")
            ),
            code=Code(local_path=parent_dir + "./train_src"),
            environment=environment,
            command="python train.py --training_data ${{inputs.training_data}} --max_epocs ${{inputs.max_epocs}} "
                    "--learning_rate ${{inputs.learning_rate}} --learning_rate_schedule ${{"
                    "inputs.learning_rate_schedule}} --model_output ${{outputs.model_output}} "
        )
        train_func = dsl.load_component(component=train_component)
        train_job = train_func(
            training_data=pipeline_job_training_input,
            max_epocs=pipeline_job_training_max_epocs,
            learning_rate=pipeline_job_training_learning_rate,
            learning_rate_schedule=pipeline_job_learning_rate_schedule,
        )

        score_component = CommandComponent(
            name="Score_Inline",   # change name temporarily because of component exist with different code
            version="10",
            inputs=dict(
                model_input=dict(type="path"),
                test_data=dict(type="path"),
            ),
            outputs=dict(
                score_output=dict(type="path")
            ),
            code=Code(local_path=parent_dir + "./score_src"),
            environment=environment,
            command="python score.py --model_input ${{inputs.model_input}} --test_data ${{inputs.test_data}} "
                    "--score_output ${{outputs.score_output}} "
        )
        score_func = dsl.load_component(component=score_component)
        score_job = score_func(
            model_input=train_job.outputs.model_output,
            test_data=pipeline_job_test_input,
        )

        eval_component = CommandComponent(
            name="Eval_Inline",
            version="10",
            inputs=dict(
                scoring_result=dict(type="path"),
            ),
            outputs=dict(
                eval_output=dict(type="path")
            ),
            code=Code(local_path=parent_dir + "./eval_src"),
            environment=environment,
            command="python eval.py --scoring_result ${{inputs.scoring_result}} --eval_output ${{outputs.eval_output}}"
        )
        eval_func = dsl.load_component(component=eval_component)
        evaluate_job = eval_func(scoring_result=score_job.outputs.score_output)
        return {
            "pipeline_job_trained_model": train_job.outputs.model_output,
            "pipeline_job_scored_data": score_job.outputs.score_output,
            "pipeline_job_evaluation_report": evaluate_job.outputs.eval_output,
        }

    pipeline = sample_pipeline(
        Dataset(local_path=parent_dir + "./data/"),
        Dataset(local_path=parent_dir + "./data/"),
        20,
        1.8,
        "time-based",
    )
    pipeline.outputs.pipeline_job_trained_model.mode = "rw_mount"
    pipeline.outputs.pipeline_job_scored_data.mode = "rw_mount"
    pipeline.outputs.pipeline_job_evaluation_report.mode = "rw_mount"
    return pipeline
