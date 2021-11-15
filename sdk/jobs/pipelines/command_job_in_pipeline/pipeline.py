from azure.ml import dsl
from azure.ml.dsl import Pipeline, Output
from azure.ml.entities import Dataset, Code, Job, CommandJob
# from azure.ml.entities import JobInputDataset
from azure.ml.entities import InputDatasetEntry
from pathlib import Path

parent_dir = str(Path(__file__).parent)


# define command jobs
environment = "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:5"

train_job = CommandJob(
    inputs=dict(
        training_data=InputDatasetEntry(dataset=Dataset(local_path=parent_dir + "./data/")),
        max_epocs=20,
        learning_rate=1.8,
        learning_rate_schedule='time-based',
        ),
    outputs=dict(
        model_output=None
        ),
    display_name="my-train-job",
    code=Code(local_path=parent_dir + "./train_src"),
    environment=environment,
    compute="cpu-cluster",
    command="""python train.py --training_data ${{inputs.training_data}} --max_epocs ${{inputs.max_epocs}}
        --learning_rate ${{inputs.learning_rate}} --learning_rate_schedule 
        ${{inputs.learning_rate_schedule}} --model_output ${{outputs.model_output}}"""
    )

score_job = CommandJob(
    inputs=dict(
        model_input=InputDatasetEntry(dataset=Dataset(local_path=parent_dir + "./data/")),
        test_data=InputDatasetEntry(dataset=Dataset(local_path=parent_dir + "./data/")),
        ),
    outputs=dict(
        score_output=None
        ),
    display_name="my-score-job",
    code=Code(local_path=parent_dir + "./score_src"),
    environment=environment,
    command="""python score.py --model_input ${{inputs.model_input}}
        --test_data ${{inputs.test_data}} --score_output ${{outputs.score_output}}"""
        )

eval_job = CommandJob(
    inputs=dict(
        scoring_result=InputDatasetEntry(dataset=Dataset(local_path=parent_dir + "./data/")),
    ),
    outputs=dict(
        eval_output=None
    ),
    display_name="my-evaluate-job",
    environment=environment,
    command='echo "hello world"'
)

def generate_dsl_pipeline() -> Pipeline:
    # 1. Construct pipeline with command job
    @dsl.pipeline(
        compute="cpu-cluster",
    )
    def sample_pipeline(
            pipeline_job_training_input,
            pipeline_job_test_input,
            pipeline_job_training_max_epocs,
            pipeline_job_training_learning_rate,
            pipeline_job_learning_rate_schedule,
    ):
        
        train_func = dsl.load_component(component=train_job)
        train_node = train_func(
            training_data=pipeline_job_training_input,
            max_epocs=pipeline_job_training_max_epocs,
            learning_rate=pipeline_job_training_learning_rate,
            learning_rate_schedule=pipeline_job_learning_rate_schedule,
        )

        score_func = dsl.load_component(component=score_job)
        score_node = score_func(
            model_input=train_node.outputs.model_output,
            test_data=pipeline_job_test_input,
        )
        eval_func = dsl.load_component(component=eval_job)
        eval_node = eval_func(scoring_result=score_node.outputs.score_output)
        return {
            "pipeline_job_trained_model": train_node.outputs.model_output,
            "pipeline_job_scored_data": score_node.outputs.score_output,
            "pipeline_job_evaluation_report": eval_node.outputs.eval_output,
        }

    pipeline = sample_pipeline(
        Dataset(local_path=parent_dir + "./data/"),
        Dataset(local_path=parent_dir + "./data/"),
        20,
        1.8,
        "time-based",
    )
    pipeline.outputs.pipeline_job_trained_model.mode = "upload"
    pipeline.outputs.pipeline_job_scored_data.mode = "upload"
    pipeline.outputs.pipeline_job_evaluation_report.mode = "upload"

    return pipeline
