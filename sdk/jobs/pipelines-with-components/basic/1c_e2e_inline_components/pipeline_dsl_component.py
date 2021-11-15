from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os

from azure.ml import dsl
from azure.ml.dsl import Pipeline
from azure.ml.entities import Dataset, Code


# We propose using dsl.command_component to support inline component in dsl.pipeline.
# Which allows user define a pipeline job without YAML.
# TODO: add this to test when we supports dsl.command_component

def generate_dsl_pipeline() -> Pipeline:
    # 1. Construct components
    environment = "azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:5"

    @dsl.command_component(
        display_name="my-train-job",
        environment=environment,
    )
    def train_func(training_data, model_output: Output = None, max_epocs=20, learning_rate=1.8, learning_rate_schedule='time-based'):
        print("hello training world...")

        lines = [
            f'Training data path: {training_data}',
            f'Max epocs: {max_epocs}',
            f'Learning rate: {learning_rate}',
            f'Learning rate: {learning_rate_schedule}',
            f'Model output path: {model_output}',
        ]

        for line in lines:
            print(line)

        print("mounted_path files: ")
        arr = os.listdir(training_data)
        print(arr)

        for filename in arr:
            print("reading file: %s ..." % filename)
            with open(os.path.join(training_data, filename), 'r') as handle:
                print(handle.read())

        # Do the train and save the trained model as a file into the output folder.
        # Here only output a dummy data for demo.
        curtime = datetime.now().strftime("%b-%d-%Y %H:%M:%S")
        model = f"This is a dummy model with id: {str(uuid4())} generated at: {curtime}\n"
        (Path(model_output) / 'model.txt').write_text(model)

    @dsl.command_component(
        display_name="my-score-job",
        environment=environment,
    )
    def score_func(model_input, test_data, score_output: Output = None):
        print("hello scoring world...")

        lines = [
            f'Model path: {model_input}',
            f'Test data path: {test_data}',
            f'Scoring output path: {score_output}',
        ]

        for line in lines:
            print(line)

        # Load the model from input port
        # Here only print the model as text since it is a dummy one
        model = (Path(model_input) / 'model.txt').read_text()
        print('Model: ', model)

        # Do scoring with the input model
        # Here only print text to output file as demo
        (Path(score_output) / 'score.txt').write_text('Scored with the following mode:\n{}'.format(model))

    @dsl.command_component(
        display_name="my-evaluate-job",
        environment=environment,
    )
    def eval_func(scoring_result, eval_output: Output = None):
        print("hello evaluation world...")

        lines = [
            f'Scoring result path: {scoring_result}',
            f'Evaluation output path: {eval_output}',
        ]

        for line in lines:
            print(line)

        # Evaluate the incoming scoring result and output evaluation result.
        # Here only output a dummy file for demo.
        curtime = datetime.now().strftime("%b-%d-%Y %H:%M:%S")
        eval_msg = f"Eval done at {curtime}\n"
        (Path(eval_output) / 'eval_result.txt').write_text(eval_msg)

    # 2. Construct pipeline
    @dsl.pipeline(
        compute="azureml:cpu-cluster",
        description="Dummy train-score-eval pipeline with local components",
    )
    def sample_pipeline(
            pipeline_job_training_input,
            pipeline_job_test_input,
            pipeline_job_training_max_epocs,
            pipeline_job_training_learning_rate,
            pipeline_job_learning_rate_schedule,
    ):
        train_job = train_func(
            training_data=pipeline_job_training_input,
            max_epocs=pipeline_job_training_max_epocs,
            learning_rate=pipeline_job_training_learning_rate,
            learning_rate_schedule=pipeline_job_learning_rate_schedule,
        )
        score_job = score_func(
            model_input=train_job.outputs.model_output,
            test_data=pipeline_job_test_input,
        )
        evalute_job = eval_func(scoring_result=score_job.outputs.score_output)
        return {
            "pipeline_job_trained_model": train_job.outputs.model_output,
            "pipeline_job_scored_data": score_job.outputs.score_output,
            "pipeline_job_evaluation_report": evalute_job.outputs.eval_output,
        }

    pipeline = sample_pipeline(
        Dataset(local_path="tests/test_configs/pipeline_samples/1c_e2e_inline_components/data/"),
        Dataset(local_path="tests/test_configs/pipeline_samples/1c_e2e_inline_components/data/"),
        20,
        1.8,
        "time-based",
    )

    return pipeline
