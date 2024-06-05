import argparse
import base64
import json
import os
import urllib
import xml.etree.ElementTree as ET

from zipfile import ZipFile

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from mot2coco import main as mot2coco_converter
from cocovid2jsonl import main as cocovid2jsonl_converter


def create_ml_table_file(filename):
    """Create ML Table definition
    :param filename: Name of the jsonl file
    """

    return (
        "paths:\n"
        "  - file: ./{0}\n"
        "transformations:\n"
        "  - read_json_lines:\n"
        "        encoding: utf8\n"
        "        invalid_lines: error\n"
        "        include_path_column: false\n"
        "  - convert_column_types:\n"
        "      - columns: image_url\n"
        "        column_type: stream_info"
    ).format(filename)


def save_ml_table_file(output_path, mltable_file_contents):
    """Save ML Table file
    :param output_path: Path to save the MLTable file
    :param mltable_file_contents: Contents of the MLTable file
    """
    with open(os.path.join(output_path, "MLTable"), "w") as f:
        f.write(mltable_file_contents)


def create_jsonl_and_mltable_files(uri_folder_data_path, dataset_dir):
    """Create jsonl

    :param uri_folder_data_path: Path to the data folder
    :param dataset_dir: Path to the dataset folder
    """
    # We'll copy each JSONL file within its related MLTable folder
    training_mltable_path = os.path.join(dataset_dir, "../training-mltable-folder")
    validation_mltable_path = os.path.join(dataset_dir, "../validation-mltable-folder")
    testing_mltable_path = os.path.join(dataset_dir, "../testing-mltable-folder")

    # First, let's create the folders if they don't exist
    os.makedirs(training_mltable_path, exist_ok=True)
    os.makedirs(validation_mltable_path, exist_ok=True)
    os.makedirs(testing_mltable_path, exist_ok=True)

    train_annotations_file = os.path.join(
        training_mltable_path, "train_annotations.jsonl"
    )
    validation_annotations_file = os.path.join(
        validation_mltable_path, "validation_annotations.jsonl"
    )
    testing_annotations_file = os.path.join(
        testing_mltable_path, "testing_annotations.jsonl"
    )

    print("Creating jsonl files")

    # Second, convert the COCO format to jsonl
    print("convert MOT format to COCO format")
    mot2coco_converter(
        argparse.Namespace(
            input=dataset_dir,
            output=f"{dataset_dir}/annotations",
            convert_det=True,
            split_train=True,
        )
    )
    print("Converting COCO video format to jsonl")
    cocovid2jsonl_converter(
        argparse.Namespace(
            input_cocovid_file_path=f"{dataset_dir}/annotations/half-train_cocoformat.json",
            output_dir=training_mltable_path,
            output_file_name="train_annotations.jsonl",
            task_type="ObjectTracking",
            base_url=f"{uri_folder_data_path}train",
        )
    )
    cocovid2jsonl_converter(
        argparse.Namespace(
            input_cocovid_file_path=f"{dataset_dir}/annotations/half-val_cocoformat.json",
            output_dir=validation_mltable_path,
            output_file_name="validation_annotations.jsonl",
            task_type="ObjectTracking",
            base_url=f"{uri_folder_data_path}train",
        )
    )

    # Create and save train mltable
    print("create and save train mltable")
    train_mltable_file_contents = create_ml_table_file(
        os.path.basename(train_annotations_file)
    )
    save_ml_table_file(training_mltable_path, train_mltable_file_contents)

    # Create and save validation mltable
    print("create and save validation mltable")
    validation_mltable_file_contents = create_ml_table_file(
        os.path.basename(validation_annotations_file)
    )
    save_ml_table_file(validation_mltable_path, validation_mltable_file_contents)

    # Create and save testing mltable
    testing_mltable_file_contents = create_ml_table_file(
        os.path.basename(testing_annotations_file)
    )
    save_ml_table_file(testing_mltable_path, testing_mltable_file_contents)


def upload_data_and_create_jsonl_mltable_files(ml_client, dataset_parent_dir):
    """upload data to blob storage and create jsonl and mltable files

    :param ml_client: Azure ML client
    :param dataset_parent_dir: Path to the dataset folder
    """
    # Change to a different location if you prefer
    dataset_parent_dir = "data"

    # create data folder if it doesnt exist.
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # download data
    download_url = "https://download.openmmlab.com/mmtracking/data/MOT17_tiny.zip"

    # Extract current dataset name from dataset url
    dataset_name = os.path.split(download_url)[-1].split(".")[0]
    # Get dataset path for later use
    dataset_dir = os.path.join(dataset_parent_dir, dataset_name)

    # Get the data zip file path
    data_file = os.path.join(dataset_parent_dir, f"{dataset_name}.zip")

    # Download the dataset
    urllib.request.urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, "r") as zzip:
        print("extracting files...")
        zzip.extractall(path=dataset_parent_dir)
        print("done")
    # delete zip file
    os.remove(data_file)

    # Upload data and create a data asset URI folder
    print("Uploading data to blob storage")
    my_data = Data(
        path=dataset_dir,
        type=AssetTypes.URI_FOLDER,
        description=f"{dataset_name} dataset folder",
        name=f"{dataset_name}_sample_folder",
    )

    uri_folder_data_asset = ml_client.data.create_or_update(my_data)

    print(uri_folder_data_asset)
    print("")
    print("Path to folder in Blob Storage:")
    print(uri_folder_data_asset.path)

    create_jsonl_and_mltable_files(
        uri_folder_data_path=uri_folder_data_asset.path, dataset_dir=dataset_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for video multi-object tracking"
    )

    parser.add_argument("--subscription", type=str, help="Subscription ID")
    parser.add_argument("--group", type=str, help="Resource group name")
    parser.add_argument("--workspace", type=str, help="Workspace name")
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Dataset location"
    )

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    credential = DefaultAzureCredential()
    ml_client = None
    subscription_id = args.subscription
    resource_group = args.group
    workspace = args.workspace
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)

    upload_data_and_create_jsonl_mltable_files(
        ml_client=ml_client, dataset_parent_dir=args.data_path
    )
