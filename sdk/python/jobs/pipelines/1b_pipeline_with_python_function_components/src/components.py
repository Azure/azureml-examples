from pathlib import Path
from random import randint
from uuid import uuid4

# mldesigner package contains the command_component which can be used to define component from a python function
from mldesigner import command_component, Input, Output


@command_component()
def train_model(
    training_data: Input(type="uri_file"),
    max_epochs: int,
    model_output: Output(type="uri_folder"),
    learning_rate=0.02,
):
    """A dummy train component.

    Args:
        training_data: a file contains training data
        max_epochs: max epochs
        learning_rate: learning rate
        model_output: target folder to save model output
    """

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


@command_component(
    display_name="Score",
    # init customer environment with conda YAML
    # the YAML file shall be put under your code folder.
    environment="./env.yaml",
    # specify your code folder, default code folder is current file's parent
    # code='.'
)
def score_data(
    model_input: Input(type="uri_folder"),
    test_data: Input(type="uri_file"),
    score_output: Output(type="uri_folder"),
):
    """A dummy score component."""

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


@command_component(display_name="Evaluate", environment="./env.yaml")
def eval_model(
    scoring_result: Input(type="uri_folder"), eval_output: Output(type="uri_folder")
):
    """A dummy evaluate component."""

    lines = [
        f"Scoring result path: {scoring_result}",
        f"Evaluation output path: {eval_output}",
    ]

    for line in lines:
        print(line)

    # Evaluate the incoming scoring result and output evaluation result.
    # Here only output a dummy file for demo.
    (Path(eval_output) / "eval_result").write_text("eval_result")
