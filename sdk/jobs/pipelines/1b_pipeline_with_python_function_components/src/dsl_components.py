from pathlib import Path
from random import randint
from uuid import uuid4

from azure.ml import dsl, Input, Output
from azure.ml.entities import Environment

# init customer environment with conda YAML
# the YAML file shall be put under your code folder.
conda_env = Environment(
    conda_file=Path(__file__).parent / "conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
)


@dsl.command_component(
    name="dsl_train_model",
    display_name="Train",
    description="A dummy train component defined by dsl component.",
    version="0.0.2",
    # specify distribution type if needed
    # distribution={'type': 'mpi'},
    # specify customer environment, note that azure-ml must be included.
    environment=conda_env,
    # specify your code folder, default code folder is current file's parent
    # code='.'
)
def train_model(
    training_data: Input,
    max_epochs: int,
    model_output: Output,
    learning_rate=0.02,
):
    lines = [
        f"Training data path: {training_data}",
        f"Max epochs: {max_epochs}",
        f"Learning rate: {learning_rate}",
        f"Model output path: {model_output}",
    ]

    for line in lines:
        print(line)

    # Do the train and save the trained model as a file into the output folder.
    # Here only output a dummy data for demo.
    model = str(uuid4())
    (Path(model_output) / "model").write_text(model)


@dsl.command_component(
    name="dsl_score_data",
    display_name="Score",
    description="A dummy score component defined by dsl component.",
    version="0.0.1",
    environment=conda_env,
)
def score_data(
    model_input: Input,
    test_data: Input,
    score_output: Output,
):

    lines = [
        f"Model path: {model_input}",
        f"Test data path: {test_data}",
        f"Scoring output path: {score_output}",
    ]

    for line in lines:
        print(line)

    # Load the model from input port
    # Here only print the model as text since it is a dummy one
    model = (Path(model_input) / "model").read_text()
    print("Model:", model)

    # Do scoring with the input model
    # Here only print text to output file as demo
    (Path(score_output) / "score").write_text("scored with {}".format(model))


@dsl.command_component(
    name="dsl_eval_model",
    display_name="Evaluate",
    description="A dummy evaluate component defined by dsl component.",
    version="0.0.1",
    environment=conda_env,
)
def eval_model(
    scoring_result: Input,
    eval_output: Output,
):
    lines = [
        f"Scoring result path: {scoring_result}",
        f"Evaluation output path: {eval_output}",
    ]

    for line in lines:
        print(line)

    # Evaluate the incoming scoring result and output evaluation result.
    # Here only output a dummy file for demo.
    (Path(eval_output) / "eval_result").write_text("eval_result")
