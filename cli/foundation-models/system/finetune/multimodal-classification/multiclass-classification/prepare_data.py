import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib
from os.path import basename, dirname, join
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


def create_jsonl_and_mltable_files(uri_folder_data_path, dataset_dir, csv_file_path):
    print("Creating jsonl files")

    dataset_parent_dir = os.path.dirname(dataset_dir)

    # We will copy each JSONL file within its related MLTable folder
    training_mltable_path = os.path.join(dataset_parent_dir, "training-mltable-folder")
    validation_mltable_path = os.path.join(
        dataset_parent_dir, "validation-mltable-folder"
    )

    # Create the folders if they don't exist
    os.makedirs(training_mltable_path, exist_ok=True)
    os.makedirs(validation_mltable_path, exist_ok=True)

    # Path to the training and validation files
    train_annotations_file = os.path.join(
        training_mltable_path, "train_annotations.jsonl"
    )
    validation_annotations_file = os.path.join(
        validation_mltable_path, "validation_annotations.jsonl"
    )

    train_validation_ratio = 0.2

    # Read a sample row from dataset
    df = pd.read_csv(csv_file_path)
    label_column_name = "room_type"
    train_df, val_df = train_test_split(
        df,
        test_size=train_validation_ratio,
        random_state=0,
        stratify=df[[label_column_name]],
    )

    # Save the DataFrame to a JSON Lines file
    train_df.to_json(train_annotations_file, orient="records", lines=True)
    val_df.to_json(validation_annotations_file, orient="records", lines=True)

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
    # Create data folder if it doesnt exist.
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # Download data
    download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/multimodal_classification/AirBnb.zip"

    # Extract current dataset name from dataset url
    dataset_name = os.path.split(download_url)[-1].split(".")[0]
    # Get the data zip file path
    data_file = os.path.join(dataset_parent_dir, f"{dataset_name}.zip")

    # Download the dataset
    urllib.request.urlretrieve(download_url, filename=data_file)

    # Extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall(path=dataset_parent_dir)
        print("done")
    # Delete zip file
    os.remove(data_file)

    # Initialize dataset specific fields
    dataset_dir = os.path.join(dataset_parent_dir, dataset_name)
    csv_file_path = os.path.join(dataset_dir, "airbnb_multiclass_dataset.csv")
    # Directory in which we have our images
    images_dir = os.path.join(dataset_dir, "room_images")
    image_column_name = "picture_url"

    # Upload data and create a data asset URI folder
    print("Uploading data to blob storage")
    my_data = Data(
        path=images_dir,
        type=AssetTypes.URI_FOLDER,
        description="AirBnb Room images",
        name="airbnb-roomtype-multimodal-multiclass-classif",
    )

    uri_folder_data_asset = ml_client.data.create_or_update(my_data)

    print(uri_folder_data_asset)
    print("")
    print("Path to folder in Blob Storage:")
    print(uri_folder_data_asset.path)

    # update the path
    update_img_url(image_column_name, uri_folder_data_asset.path, csv_file_path)
    print("local path replaced with AML path")

    create_jsonl_and_mltable_files(
        uri_folder_data_path=uri_folder_data_asset.path,
        dataset_dir=dataset_dir,
        csv_file_path=csv_file_path,
    )


def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


def update_img_url(img_col_name: str, image_url_prefix: str, file_name: str):
    """
    Load .csv file at path `file_name`,
    extract file name of image from path in column `img_col_name`,
    add `image_url_prefix` to the file name and update the column `img_col_name`.

    :param img_col_name:Column name in csv file that has path to images.
    :type img_col_name: str
    :param image_url_prefix: URL of datastore where images are uploaded.
    :type image_url_prefix: str
    :param file_name: .csv file that has the data.
    :type file_name: str

    :return: None
    """
    df = pd.read_csv(file_name)
    df[img_col_name] = df[img_col_name].apply(
        lambda x: join(image_url_prefix + join(basename(dirname(x)), basename(x)))
    )
    df.to_csv(file_name, index=False)


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
