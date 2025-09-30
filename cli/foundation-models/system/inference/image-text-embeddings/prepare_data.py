import argparse
import base64
import json
import os
import shutil
import urllib.request
import pandas as pd
from zipfile import ZipFile
import random
import string


def download_and_unzip(dataset_parent_dir: str) -> None:
    """Download image dataset and unzip it.

    :param dataset_parent_dir: dataset parent directory to which dataset will be downloaded
    :type dataset_parent_dir: str
    """
    # Create directory, if it does not exist
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # download data
    download_url = "https://automlsamplenotebookdata-adcuc7f7bqhhh8a4.b02.azurefd.net/image-classification/fridgeObjects.zip"
    print(f"Downloading data from {download_url}")

    # Extract current dataset name from dataset url
    dataset_name = os.path.basename(download_url).split(".")[0]
    # Get dataset path for later use
    dataset_dir = os.path.join(dataset_parent_dir, dataset_name)

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Get the name of zip file
    data_file = os.path.join(dataset_parent_dir, f"{dataset_name}.zip")

    # Download data from public url
    urllib.request.urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall(path=dataset_parent_dir)
        print("done")
    # delete zip file
    os.remove(data_file)
    return dataset_dir


def read_image(image_path: str) -> bytes:
    """Read image from path.

    :param image_path: image path
    :type image_path: str
    :return: image in bytes format
    :rtype: bytes
    """
    with open(image_path, "rb") as f:
        return f.read()


def prepare_data_for_online_inference(dataset_dir: str) -> None:
    """Prepare request json files for online inference.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    """
    sample_image_1 = os.path.join(dataset_dir, "milk_bottle", "99.jpg")
    sample_image_2 = os.path.join(dataset_dir, "can", "1.jpg")

    # Generate sample request for image embeddings
    image_request_json = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0, 1],
            "data": [
                [
                    base64.encodebytes(read_image(sample_image_1)).decode("utf-8"),
                    "",
                ],  # the "text" column should contain empty string
                [base64.encodebytes(read_image(sample_image_2)).decode("utf-8"), ""],
            ],
        }
    }

    request_file_name = os.path.join(dataset_dir, "image_request_data.json")

    with open(request_file_name, "w") as request_file:
        json.dump(image_request_json, request_file)

    # Generate sample request for text embeddings
    text_request_json = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0, 1],
            "data": [
                [
                    "",
                    "a photo of a milk bottle",
                ],  # the "image" column should contain empty string
                ["", "a photo of a metal can"],
            ],
        }
    }

    request_file_name = os.path.join(dataset_dir, "text_request_data.json")

    with open(request_file_name, "w") as request_file:
        json.dump(text_request_json, request_file)

    # Generate sample request for image and text embeddings
    image_text_request_json = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0, 1],
            "data": [
                [
                    base64.encodebytes(read_image(sample_image_1)).decode("utf-8"),
                    "a photo of a milk bottle",
                ],  # all rows should have both images and text
                [
                    base64.encodebytes(read_image(sample_image_2)).decode("utf-8"),
                    "a photo of a metal can",
                ],
            ],
        }
    }

    request_file_name = os.path.join(dataset_dir, "image_text_request_data.json")

    with open(request_file_name, "w") as request_file:
        json.dump(image_text_request_json, request_file)


def prepare_data_for_batch_inference(dataset_dir: str) -> None:
    """Prepare image folder and csv files for batch inference.

    This function will move all images to a single image folder and also create folders of csv
    files. Each folder will have csv files that contain images in base64 format, text samples, or both.
    :param dataset_dir: dataset directory
    :type dataset_dir: str
    """
    batch_input_file = "batch_input.csv"
    # Generate batch input for image embeddings
    image_list = []
    image_path_list = []

    for dir_name in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, dir_name)
        for path, _, files in os.walk(dir_path):
            for file in files:
                image_path = os.path.join(path, file)
                image = read_image(image_path)
                image_path_list.append(image_path)
                image_list.append(base64.encodebytes(image).decode("utf-8"))

    image_data = [[image, ""] for image in image_list]
    batch_df = pd.DataFrame(image_data, columns=["image", "text"])

    image_csv_folder_path = os.path.join(dataset_dir, "image_batch")
    os.makedirs(image_csv_folder_path, exist_ok=True)
    # Divide this into files of 10 rows each
    batch_size_per_predict = 10
    for i in range(0, len(batch_df), batch_size_per_predict):
        j = i + batch_size_per_predict
        batch_df[i:j].to_csv(
            os.path.join(image_csv_folder_path, str(i) + batch_input_file)
        )

    # Generate batch input for text embeddings
    # supply strings describing the images
    text_data = [
        ["", "a photo of a " + os.path.basename(os.path.dirname(image_path))]
        for image_path in image_path_list
    ]
    batch_df = pd.DataFrame(text_data, columns=["image", "text"])

    text_csv_folder_path = os.path.join(dataset_dir, "text_batch")
    os.makedirs(text_csv_folder_path, exist_ok=True)
    # Divide this into files of 10 rows each
    batch_size_per_predict = 10
    for i in range(0, len(batch_df), batch_size_per_predict):
        j = i + batch_size_per_predict
        batch_df[i:j].to_csv(
            os.path.join(text_csv_folder_path, str(i) + batch_input_file)
        )

    # Generate batch input for image and text embeddings
    # supply base64 images for images samples and random strings for text samples
    image_text_data = [
        [image_list[i], "a photo of a " + os.path.basename(os.path.dirname(image_path))]
        for i in range(len(image_list))
    ]
    batch_df = pd.DataFrame(image_text_data, columns=["image", "text"])

    image_text_csv_folder_path = os.path.join(dataset_dir, "image_text_batch")
    os.makedirs(image_text_csv_folder_path, exist_ok=True)
    # Divide this into files of 10 rows each
    batch_size_per_predict = 10
    for i in range(0, len(batch_df), batch_size_per_predict):
        j = i + batch_size_per_predict
        batch_df[i:j].to_csv(
            os.path.join(image_text_csv_folder_path, str(i) + batch_input_file)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for zero-shot image classification"
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="Dataset location"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        help="prepare data for online or batch inference",
    )

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    dataset_dir = download_and_unzip(
        dataset_parent_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), args.data_path
        ),
    )

    if args.mode == "online":
        prepare_data_for_online_inference(dataset_dir=dataset_dir)
    else:
        prepare_data_for_batch_inference(dataset_dir=dataset_dir)
