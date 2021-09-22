"""
Basic evaluation step to measure the accuracy of a trained model

In hyperparameter tuning, the evaluation needs to be made using the 
best model selected by the HyperDrive step.
"""

import logging
from os.path import join
import json
from shutil import copy

import click
from azureml.core import Run


@click.command()
@click.option("--root_dir", type=click.STRING, required=True,
              help="Root directory of datastore")
@click.option("--model_info_dir", type=click.STRING, required=True,
              help="Path to folder to save trained model information")
@click.option("--model_info_best", type=str, required=True,
              help="Path to an archive from hyperdrive step")
def main(root_dir: str,
         model_info_dir: str,
         model_info_best: str) -> None:

    """
    Main function for receiving args, and passing them through
    to basic training step

    root_dir: str
        Root directory of datastore being used
    model_info_dir: str
        Path to folder to save trained model information
    model_info_best: str
        Path to best model in archive from hyperdrive step
    """

    logging.basicConfig(level=logging.INFO)
    log: logging.Logger = logging.getLogger(__name__)
    log.info("Hyperdrive Evaluation Step")

    # Get context of the current run
    run = Run.get_context()

    # Compile directories
    model_info_dir = join(root_dir, model_info_dir)
    model_info_best = join(root_dir, model_info_best)

    # Copy info of the best model to model directory
    path_to_model = join(model_info_best, "outputs/model/model.json")
    copy(path_to_model, model_info_dir)

    # Compile model information
    log.info("Compile model information")
    model_fname = "model.json"
    with open(join(model_info_dir, model_fname), "r") as model_info_file:
        model_info = json.load(model_info_file)

    # Calculate fake accuracy score for evaluation step
    initial_lr = model_info["initial_lr"]
    accuracy = (1/initial_lr) * model_info["accuracy"]
    accuracy = accuracy / 100

    # Log metrics to AML
    run.log(name="accuracy", value=accuracy)
    run.log(name="best_learning_rate", value=initial_lr)

    log.info("Evaluation step has been completed")
    return


if __name__ == "__main__":
    main()
