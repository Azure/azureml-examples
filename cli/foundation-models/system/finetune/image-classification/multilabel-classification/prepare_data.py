import argparse
import base64
import json
import os
import urllib
from zipfile import ZipFile

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


def create_ml_table_file(filename):
    """Create ML Table definition"""

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
    with open(os.path.join(output_path, "MLTable"), "w") as f:
        f.write(mltable_file_contents)


def create_jsonl_and_mltable_files(uri_folder_data_path, dataset_dir):
    print("Creating jsonl files")

    dataset_parent_dir = os.path.dirname(dataset_dir)

    # We will copy each JSONL file within its related MLTable folder
    training_mltable_path = os.path.join(dataset_parent_dir, "training-mltable-folder")
    validation_mltable_path = os.path.join(
        dataset_parent_dir, "validation-mltable-folder"
    )

    # Create MLTable folders, if they don't exist
    os.makedirs(training_mltable_path, exist_ok=True)
    os.makedirs(validation_mltable_path, exist_ok=True)

    train_validation_ratio = 5

    # Path to the training and validation files
    train_annotations_file = os.path.join(
        training_mltable_path, "train_annotations.jsonl"
    )
    validation_annotations_file = os.path.join(
        validation_mltable_path, "validation_annotations.jsonl"
    )

    # Path to the labels file.
    label_file = os.path.join(dataset_dir, "labels.csv")

    # Baseline of json line dictionary
    json_line_sample = {"image_url": uri_folder_data_path, "label": ""}

    index = 0
    # Read each annotation and convert it to jsonl line
    with open(train_annotations_file, "w") as train_f:
        with open(validation_annotations_file, "w") as validation_f:
            with open(label_file, "r") as labels:
                for i, line in enumerate(labels):
                    # Skipping the title line and any empty lines.
                    if i == 0 or len(line.strip()) == 0:
                        continue
                    line_split = line.strip().split(",")
                    if len(line_split) != 2:
                        print("Skipping the invalid line: {}".format(line))
                        continue
                    json_line = dict(json_line_sample)
                    json_line["image_url"] += f"images/{line_split[0]}"
                    json_line["label"] = line_split[1].strip().split(" ")

                    if i % train_validation_ratio == 0:
                        # Validation annotation
                        validation_f.write(json.dumps(json_line) + "\n")
                    else:
                        # Train annotation
                        train_f.write(json.dumps(json_line) + "\n")
    print("done")

    # Create and save train mltable
    train_mltable_file_contents = create_ml_table_file(
        os.path.basename(train_annotations_file)
    )
    save_ml_table_file(training_mltable_path, train_mltable_file_contents)

    # Create and save validation mltable
    validation_mltable_file_contents = create_ml_table_file(
        os.path.basename(validation_annotations_file)
    )
    save_ml_table_file(validation_mltable_path, validation_mltable_file_contents)


def upload_data_and_create_jsonl_mltable_files(ml_client, dataset_parent_dir):
    # Create directory, if it does not exist
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # Download data
    print("Downloading data.")
    download_url = "https://automlsamplenotebookdata-adcuc7f7bqhhh8a4.b02.azurefd.net/image-classification/multilabelFridgeObjects.zip"

    # Extract current dataset name from dataset url
    dataset_name = os.path.basename(download_url).split(".")[0]
    # Get dataset path for later use
    dataset_dir = os.path.join(dataset_parent_dir, dataset_name)

    # Get the name of zip file
    data_file = os.path.join(dataset_parent_dir, f"{dataset_name}.zip")

    # Download data from public url
    urllib.request.urlretrieve(download_url, filename=data_file)

    # Extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall(path=dataset_parent_dir)
        print("done")
    # Delete zip file
    os.remove(data_file)

    # Upload data and create a data asset URI folder
    print("Uploading data to blob storage")
    my_data = Data(
        path=dataset_dir,
        type=AssetTypes.URI_FOLDER,
        description="Fridge-items images",
        name="fridge-items-images-ml-ft",
    )

    uri_folder_data_asset = ml_client.data.create_or_update(my_data)

    print(uri_folder_data_asset)
    print("")
    print("Path to folder in Blob Storage:")
    print(uri_folder_data_asset.path)
    create_jsonl_and_mltable_files(
        uri_folder_data_path=uri_folder_data_asset.path, dataset_dir=dataset_dir
    )


def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for image classification"
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

    sample_image = os.path.join(
        args.data_path, "multilabelFridgeObjects", "images", "56.jpg"
    )
    huggingface_request_json = {
        "input_data": [base64.b64encode(read_image(sample_image)).decode("utf-8")],
    }
    huggingface_request_file_name = "huggingface_sample_request_data.json"
    with open(huggingface_request_file_name, "w") as huggingface_request_file:
        json.dump(huggingface_request_json, huggingface_request_file)
