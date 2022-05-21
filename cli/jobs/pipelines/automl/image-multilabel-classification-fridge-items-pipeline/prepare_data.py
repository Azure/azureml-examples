import argparse
import json
import os
import urllib
from zipfile import ZipFile

from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


def create_jsonl_files(uri_folder_data_path):
    print("Creating jsonl files")
    src_images = "./data/multilabelFridgeObjects/"

    # We'll copy each JSONL file within its related MLTable folder
    training_mltable_path = "./data/training-mltable-folder/"
    validation_mltable_path = "./data/validation-mltable-folder/"

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
        "label": [],
    }

    # Path to the labels file.
    labelFile = os.path.join(src_images, "labels.csv")

    # Read each annotation and convert it to jsonl line
    with open(train_annotations_file, "w") as train_f:
        with open(validation_annotations_file, "w") as validation_f:
            with open(labelFile, "r") as labels:
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
                        # validation annotation
                        validation_f.write(json.dumps(json_line) + "\n")
                    else:
                        # train annotation
                        train_f.write(json.dumps(json_line) + "\n")
    print("done")


def upload_data_and_create_jsonl_files(ml_client):
    # Download data from public url

    # download data
    download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip"
    data_file = "./data/multilabelFridgeObjects.zip"
    urllib.request.urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall(path="./data")
        print("done")
    # delete zip file
    os.remove(data_file)

    # Upload data and create a data asset URI folder
    print("Uploading data to blob storage")
    my_data = Data(
        path="./data/multilabelFridgeObjects",
        type=AssetTypes.URI_FOLDER,
        description="Fridge-items images multilabel",
        name="fridge-items-images-multilabel",
    )

    uri_folder_data_asset = ml_client.data.create_or_update(my_data)

    print(uri_folder_data_asset)
    print("")
    print("Path to folder in Blob Storage:")
    print(uri_folder_data_asset.path)

    create_jsonl_files(uri_folder_data_path=uri_folder_data_asset.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for image classification"
    )

    parser.add_argument("--subscription", type=str, help="Subscription ID")
    parser.add_argument("--group", type=str, help="Resource group name")
    parser.add_argument("--workspace", type=str, help="Workspace name")

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

    upload_data_and_create_jsonl_files(ml_client=ml_client)
