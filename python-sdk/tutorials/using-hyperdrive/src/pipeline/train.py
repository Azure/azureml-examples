"""
Basic training step to run a training script on a compute target

In hyperparameter tuning, many experiments need to run with 
different combinations of hyperparameters to compare the results. 
So, this script needs to save several models resulting from 
the combination of different tuning parameters.
This is done by defining an out_dir that saves the resulting 
model from each run created by the HyperDriveStep. 
"""
import logging
import os
from os.path import join

import click
from azureml.core import Run
from src.model.algorithms import train_fictitious_model
from src.common.model_helpers import write_model_info


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
    "--thr",
    type=click.FLOAT,
    required=True,
    default=0.5,
    help="Initial Threshold used to simulate training",
)
@click.option(
    "--epoch",
    type=click.INT,
    required=True,
    default=50,
    help="Initial Learning Rate used to simulate training",
)
@click.option(
    "--lr",
    type=click.FLOAT,
    required=True,
    default=0.1,
    help="Initial Learning Rate used to simulate training",
)
@click.option(
    "--model_info_best",
    type=str,
    required=True,
    help="Path to an archive from hyperdrive step",
)
def main(
    root_dir: str,
    model_info_dir: str,
    thr: float,
    epoch: int,
    lr: float,
    model_info_best: str,
) -> None:
    """
    Main function for receiving args, and passing them through to a training step
    Parameters:
      root_dir: str: root directory of datastore being used
      model_info_dir: str: path to folder to save trained model information
      initial_lr: float: initial Learning Rate used to simulate training
      model_info_best: str: path to best model in archive from hyperdrive step
    """

    logging.basicConfig(level=logging.INFO)
    log: logging.Logger = logging.getLogger(__name__)
    log.info("Hyperdrive Training Step")

    # Get context of the current run
    run = Run.get_context()

    # Compile directories
    model_info_dir = join(root_dir, model_info_dir)
    model_info_best = join(root_dir, model_info_best)

    # Run training step
    model_info = train_fictitious_model(thr, epoch, lr)

    # Prepare a model for registration
    os.makedirs(model_info_dir, exist_ok=True)
    model_filename = "model.json"

    # Log metrics to AML
    run.log(name="precision", value=model_info["precision"])
    run.log(name="recall", value=model_info["recall"])
    run.log(name="accuracy", value=model_info["accuracy"])
    run.log(name="loss", value=model_info["loss"])
    run.log(name="F_1", value=model_info["F_1"])

    # Define a directory to save several models resulting from the
    #   combination of different tuning parameters
    out_dir = "./outputs/model"
    out_dir = join(model_info_best, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save best model
    write_model_info(out_dir, model_filename, model_info)

    log.info("Training step has been completed")
    return


if __name__ == "__main__":
    main()
