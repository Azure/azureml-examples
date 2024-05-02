import argparse
import base64
import json
import os
import subprocess

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes






 

def upload_data_and_create_jsonl_mltable_files(ml_client, dataset_parent_dir):
    # Create directory, if it does not exist
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # Download data
    print("Downloading data.")
    #if os.path.isdir(dataset_parent_dir) == False:
       # !git clone https://huggingface.co/datasets/diffusers/dog-example
       # Define the shell command to clone the Git repository
    git_command = ["git", "clone", "https://huggingface.co/datasets/diffusers/dog-example"]
    # Execute the shell command using subprocess
    try:
       subprocess.run(git_command, check=True)
       print("Git clone successful")
    except subprocess.CalledProcessError as e:
        print("Git clone failed:", e)

    # Upload data and create a data asset URI folder
    print("Uploading data to blob storage")
    my_data = Data(
        path=dataset_parent_dir,
        type=AssetTypes.URI_FOLDER,
        description="Dog images for text to image dreambooth training",
        name="dog-images-text-to-image",
    )

    uri_folder_data_asset = ml_client.data.create_or_update(my_data)

    print(uri_folder_data_asset)
    print("---------uri_folder_data_asset--------")
    print("Path to folder in Blob Storage:")
    print(uri_folder_data_asset.path)
    return uri_folder_data_asset.path

def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for text-to-image"
    )

    parser.add_argument("--subscription", type=str, help="Subscription ID")
    parser.add_argument("--group", type=str, help="Resource group name")
    parser.add_argument("--workspace", type=str, help="Workspace name")
    parser.add_argument(
        "--data_path", type=str, default="./dog-example", help="Dataset location"
    )

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    credential = DefaultAzureCredential()
    ml_client = None
    subscription_id = args.subscription
    resource_group = args.group
    workspace = args.workspace
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)

    blobpath=upload_data_and_create_jsonl_mltable_files(
        ml_client=ml_client, dataset_parent_dir=args.data_path
    )
    dataset_dir = "dog-example"
    files = os.listdir(dataset_dir)
    image_file = [file for file in files if file.endswith((".jpg", ".jpeg", ".png"))][0]
    sample_image = os.path.join(
        args.data_path, image_file
    )
    huggingface_request_json = {
        "input_data": [base64.b64encode(read_image(sample_image)).decode("utf-8")],
    }
    huggingface_request_file_name = "huggingface_sample_request_data.json"
    with open(huggingface_request_file_name, "w") as huggingface_request_file:
        json.dump(huggingface_request_json, huggingface_request_file)
