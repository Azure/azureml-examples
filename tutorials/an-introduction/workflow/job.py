from azureml.core import Workspace, Experiment
from azureml.core import Environment, Dataset, RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

# get workspace
ws = Workspace.from_config()

# create experiment
exp = Experiment(
    workspace=ws,
    name="an-introduction-pipeline"
)

# set environment based on requirements file
env = Environment.from_pip_requirements(
    name="my_env",
    file_path="./environments/requirements.txt"
)

# define run configuration
rc = RunConfiguration()
rc.environment = env

# define input dataset for data prep
ds = Dataset.File.from_files("https://azuremlexamples.blob.core.windows.net/datasets/iris.csv").as_mount()

# define output from data prep step
processed_data = PipelineData("processed_data", ws.get_default_datastore())

# data prep step
data_prep_args = [
    "--input-data", ds,
    "--output-path", processed_data
]
data_prep = PythonScriptStep(
    name="Data Prep",
    source_directory="src",
    script_name="data-prep.py",
    arguments=data_prep_args,
    inputs=[ds],
    outputs=[processed_data],
    compute_target="cpu-cluster",
    runconfig=rc
)

# training step
train_args = ["--processed-data-path", processed_data]
train = PythonScriptStep(
    name="Train Light GBM Model",
    source_directory="src",
    script_name="train.py",
    arguments=train_args,
    inputs=[processed_data],
    compute_target="cpu-cluster",
    runconfig=rc
)

# define pipeline
my_pipeline = Pipeline(ws, steps=[data_prep, train])

# submit pipelines
run = exp.submit(my_pipeline)
run.wait_for_completion()


