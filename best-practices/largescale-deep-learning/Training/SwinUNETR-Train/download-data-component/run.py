import sys
import os
import zipfile
import json
import shutil
import argparse
import logging
import urllib.request

import synapseclient
import synapseutils

from tqdm import tqdm
from azureml.core import Run, Workspace
from azureml.core.keyvault import Keyvault


def get_arg_parser(parser=None):
    """Parse the command line arguments for merge using argparse.

    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser):
        an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the component
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to store data"
    )
    return parser


def get_key_vault() -> Keyvault:
    """Retreives keyvault from current run"""
    run: Run = Run.get_context()
    logging.info(f"Got run context: {run}")
    workspace: Workspace = run.experiment.workspace
    return workspace.get_default_keyvault()


def remove_dir_contents(dir):
    import shutil

    for filename in tqdm(os.listdir(dir)):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def main():
    """Component main function.

    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """

    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args()
    remove_dir_contents(args.data_dir)

    metadata_path = os.path.join(args.data_dir, "dataset_0.json")
    with urllib.request.urlopen(
        "https://drive.google.com/uc?export=download&id=1qcGh41p-rI3H_sQ0JwOAhNiQSXriQqGi"
    ) as req:
        with open(metadata_path, "wb") as f:
            f.write(req.read())

    with urllib.request.urlopen(
        "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
    ) as req:
        with open(os.path.join(args.data_dir, "model_swinvit.pt"), "wb") as f:
            f.write(req.read())

    kv = get_key_vault()
    synapse_username = kv.get_secret("synapse-username")
    synapse_password = kv.get_secret("synapse-password")

    syn = synapseclient.Synapse()
    syn.login(synapse_username, synapse_password)
    
    # download the dataset
    dataset = syn.get(
        entity="syn3379050", downloadLocation="./temp_dataset", ifcollision="keep.local"
    )

    with zipfile.ZipFile(dataset.path, "r") as zip_ref:
        zip_ref.extractall("./temp_unzipped_dataset")

    os.makedirs(os.path.join(args.data_dir, "imagesTr"), exist_ok=True)
    for file in tqdm(os.listdir("./temp_unzipped_dataset/RawData/Training/img")):
        shutil.copyfile(
            os.path.join("./temp_unzipped_dataset/RawData/Training/img", file),
            os.path.join(args.data_dir, "imagesTr", file),
        )

    os.makedirs(os.path.join(args.data_dir, "labelsTr"), exist_ok=True)
    for file in tqdm(os.listdir("./temp_unzipped_dataset/RawData/Training/label")):
        shutil.copyfile(
            os.path.join("./temp_unzipped_dataset/RawData/Training/label", file),
            os.path.join(args.data_dir, "labelsTr", file),
        )

    os.makedirs(os.path.join(args.data_dir, "imagesTs"), exist_ok=True)
    for file in tqdm(os.listdir("./temp_unzipped_dataset/RawData/Testing/img")):
        shutil.copyfile(
            os.path.join("./temp_unzipped_dataset/RawData/Testing/img", file),
            os.path.join(args.data_dir, "imagesTs", file),
        )


if __name__ == "__main__":
    # Set logging to sys.out
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    main()
