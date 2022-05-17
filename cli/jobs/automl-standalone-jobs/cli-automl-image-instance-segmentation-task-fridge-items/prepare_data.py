import argparse
import os
import sys
import subprocess
import urllib

from zipfile import ZipFile

from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


def upload_data_and_create_jsonl_files(ml_client):
    # Download data from public url

    # download data
    download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjectsMask.zip"
    data_file = "./data/odFridgeObjectsMask.zip"
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
        path="./data/odFridgeObjectsMask",
        type=AssetTypes.URI_FOLDER,
        description="Fridge-items images instance segmentation",
        name="fridge-items-images-instance-segmentation",
    )

    uri_folder_data_asset = ml_client.data.create_or_update(my_data)

    print(uri_folder_data_asset)
    print("")
    print("Path to folder in Blob Storage:")
    print(uri_folder_data_asset.path)

    print("Installing scikit-image and simplification package")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "scikit-image==0.17.2"]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "simplification"])
    print("done")

    print("Creating jsonl files")
    from jsonl_converter import convert_mask_in_VOC_to_jsonl

    data_path = "./data/odFridgeObjectsMask/"
    convert_mask_in_VOC_to_jsonl(data_path, uri_folder_data_asset.path)
    print("done")


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
