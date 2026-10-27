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

import numpy as np


@click.command()
@click.option(
    "--root_dir", type=click.STRING, required=True, help="Root directory of datastore"
)
@click.option(
    "--model_info_dir",
    type=click.STRING,
    required=True,
    help="Path to folder to save trained model information",
)
@click.option(
    "--model_info_best",
    type=str,
    required=True,
    help="Path to an archive from hyperdrive step",
)
def main(root_dir: str, model_info_dir: str, model_info_best: str) -> None:
    """
    Main function for receiving args, and passing them through to evaluation step
    Parameters:
      root_dir: str: root directory of datastore being used
      model_info_dir: str: path to folder to save trained model information
      model_info_best: str: ath to best model in archive from hyperdrive step
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

    # Calculate performance metrics for evaluation step
    thr = model_info["thr"]
    epoch = model_info["epoch"]
    lr = model_info["lr"]

    k_a = 1
    k_p = 1
    k_r = 1
    k_l = 1

    precision_thr = 0.1 + 0.9 / (1 + 5000 * np.exp(-20 * thr))
    recall_thr = 1 / (1 + 1e-5 * np.exp(20 * thr))

    loss_epoch = 3 + 7 * np.exp(-0.05 * epoch)
    accuracy_epoch = 0.95 - 3 * (np.log10(3 - 0.025 * epoch)) ** 2

    z = np.log10(lr)
    loss_lr = 0.6 + (np.log10(0.9 - z)) ** 2
    accuracy_lr = 0.98 - (np.log10(0.7 - 0.5 * z)) ** 2

    accuracy = k_a * accuracy_epoch * accuracy_epoch * accuracy_lr
    precision = k_p * precision_thr * accuracy_epoch * accuracy_lr
    recall = k_r * recall_thr * accuracy_epoch * accuracy_lr
    loss = k_l * loss_epoch * loss_lr
    F_1 = 2 * (precision * recall) / (precision + recall)

    # Log metrics to AML
    run.log(name="accuracy", value=accuracy)
    run.log(name="precision", value=precision)
    run.log(name="recall", value=recall)
    run.log(name="loss", value=loss)
    run.log(name="F_1", value=F_1)
    run.log(name="best_threshold", value=thr)
    run.log(name="best_epoch", value=epoch)
    run.log(name="best_learning_rate", value=lr)

    log.info("Evaluation step has been completed")
    return


if __name__ == "__main__":
    main()
