"""
Basic step for a registering a model in a workspace
"""
import logging
from os.path import join
import json
import click
from azureml.core import Run
from src.common.model_helpers import write_model_info


@click.command()
@click.option(
    "--root_dir", type=click.STRING, required=True, help="Root directory of datastore"
)
@click.option(
    "--model_name",
    type=click.STRING,
    required=True,
    help="Name of model that will be registered to AML workspace",
)
@click.option(
    "--model_info_dir",
    type=click.STRING,
    required=True,
    help="Path to folder with saved trained model information",
)
def main(root_dir: str, model_name: str, model_info_dir: str) -> None:
    """
    Main function for receiving args, and passing them through to registration step
    Parameters:
      root_dir: str: root directory of datastore being used
      model_name: str: name of the model that will be registered to the AML workspace
      model_info_dir: str: path to folder with saved trained model information
    """

    logging.basicConfig(level=logging.INFO)
    log: logging.Logger = logging.getLogger(__name__)
    log.info("Basic Registration Step")

    # Compile model info directory
    model_info_dir = join(root_dir, model_info_dir)

    # Compile model information
    log.info("Compile model information")
    model_fname = "model.json"
    with open(join(model_info_dir, model_fname), "r") as model_info_file:
        model_info = json.load(model_info_file)

    # Save produced artifacts
    model_dir = "./outputs/model"
    write_model_info(model_dir, "model.json", model_info)

    # Registering everything as a model
    log.info("Register model")

    # Get context of current run and upload results
    run = Run.get_context()
    run.upload_folder(model_dir, model_dir)
    model = run.register_model(model_name=model_name, model_path=model_dir)

    log.info("Model has been registered")
    return model


if __name__ == "__main__":
    main()
