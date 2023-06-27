import argparse
import json
import os
import urllib
import xml.etree.ElementTree as ET

from zipfile import ZipFile

from azure.identity import InteractiveBrowserCredential
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

    # We'll copy each JSONL file within its related MLTable folder
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

    # Baseline of json line dictionary
    json_line_sample = {
        "image_url": uri_folder_data_path,
        "image_details": {"format": None, "width": None, "height": None},
        "label": [],
    }

    # Path to the annotations
    annotations_folder = os.path.join(dataset_dir, "annotations")

    # Read each annotation and convert it to jsonl line
    with open(train_annotations_file, "w") as train_f:
        with open(validation_annotations_file, "w") as validation_f:
            for i, filename in enumerate(os.listdir(annotations_folder)):
                if not filename.endswith(".xml"):
                    print("Skipping unknown file: {}".format(filename))
                    continue

                annotations_file_path = os.path.join(annotations_folder, filename)
                print(f"Parsing {os.path.join(annotations_folder, filename)}")

                root = ET.parse(annotations_file_path).getroot()

                width = int(root.find("size/width").text)
                height = int(root.find("size/height").text)

                labels = []
                for object in root.findall("object"):
                    name = object.find("name").text
                    xmin = object.find("bndbox/xmin").text
                    ymin = object.find("bndbox/ymin").text
                    xmax = object.find("bndbox/xmax").text
                    ymax = object.find("bndbox/ymax").text
                    isCrowd = int(object.find("difficult").text)
                    labels.append(
                        {
                            "label": name,
                            "topX": float(xmin) / width,
                            "topY": float(ymin) / height,
                            "bottomX": float(xmax) / width,
                            "bottomY": float(ymax) / height,
                            "isCrowd": isCrowd,
                        }
                    )
                # build the jsonl file
                image_filename = root.find("filename").text
                _, file_extension = os.path.splitext(image_filename)
                json_line = dict(json_line_sample)
                json_line["image_url"] = (
                    json_line["image_url"] + "images/" + image_filename
                )
                json_line["image_details"]["format"] = file_extension[1:]
                json_line["image_details"]["width"] = width
                json_line["image_details"]["height"] = height
                json_line["label"] = labels

                if i % train_validation_ratio == 0:
                    # validation annotation
                    validation_f.write(json.dumps(json_line) + "\n")
                else:
                    # train annotation
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
    # Download data from public url

    # create data folder if it doesnt exist.
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # download data
    download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"

    # Extract current dataset name from dataset url
    dataset_name = os.path.basename(download_url).split(".")[0]
    # Get dataset path for later use
    dataset_dir = os.path.join(dataset_parent_dir, dataset_name)

    # Get the data zip file path
    data_file = os.path.join(dataset_parent_dir, f"{dataset_name}.zip")

    # Download the dataset
    urllib.request.urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall(path=dataset_parent_dir)
        print("done")
    # delete zip file
    os.remove(data_file)

    # Upload data and create a data asset URI folder
    print("Uploading data to blob storage")
    my_data = Data(
        path=dataset_dir,
        type=AssetTypes.URI_FOLDER,
        description="Fridge-items images Object detection",
        name="fridge-items-images-object-detection",
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

    credential = InteractiveBrowserCredential()
    ml_client = None
    try:
        ml_client = MLClient.from_config(credential)
    except Exception as ex:
        # Enter details of your AML workspace
        subscription_id = args.subscription
        resource_group = args.group
        workspace = args.workspace
        ml_client = MLClient(credential, subscription_id, resource_group, workspace)

    upload_data_and_create_jsonl_mltable_files(
        ml_client=ml_client, dataset_parent_dir=args.data_path
    )
