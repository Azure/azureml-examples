import argparse
import base64
import json
import os
import shutil
import urllib.request
import pandas as pd
from zipfile import ZipFile

# Change this to match the inference dataset
LABELS = "water_bottle, milk_bottle, carton, can"


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
    """Prepare request json for online inference.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    """
    sample_image_1 = os.path.join(dataset_dir, "milk_bottle", "99.jpg")
    sample_image_2 = os.path.join(dataset_dir, "can", "1.jpg")

    request_json = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0, 1],
            "data": [
                [
                    base64.encodebytes(read_image(sample_image_1)).decode("utf-8"),
                    LABELS,
                ],  # the labels are required in the first row of the "text" column
                [
                    base64.encodebytes(read_image(sample_image_2)).decode("utf-8")
                ],  # all other rows in the "text" column are ignored
            ],
        }
    }

    request_file_name = os.path.join(dataset_dir, "sample_request_data.json")

    with open(request_file_name, "w") as request_file:
        json.dump(request_json, request_file)


def prepare_data_for_batch_inference(dataset_dir: str) -> None:
    """Prepare image folder and csv file for batch inference.

    This function will move all images to a single image folder and also create a folder of csv
    files with images in base64 format and the candidate labels.
    :param dataset_dir: dataset directory
    :type dataset_dir: str
    """
    image_list = []

    dir_names = []
    for dir_name in os.listdir(dataset_dir):
        dir_names.append(dir_name)
        dir_path = os.path.join(dataset_dir, dir_name)
        for path, _, files in os.walk(dir_path):
            for file in files:
                image = read_image(os.path.join(path, file))
                image_list.append(base64.encodebytes(image).decode("utf-8"))
                shutil.move(os.path.join(path, file), dataset_dir)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        else:
            os.remove(dir_path)

    # labels are only added to the first row
    # all other rows in the "text" column are ignored
    labels = ",".join(dir_names)
    data = [[image, ""] for image in image_list]
    batch_df = pd.DataFrame(data, columns=["image", "text"])

    csv_folder_path = os.path.join(dataset_dir, "batch")
    os.makedirs(csv_folder_path, exist_ok=True)
    batch_input_file = "batch_input.csv"
    # Divide this into files of 10 rows each
    batch_size_per_predict = 10
    for i in range(0, len(batch_df), batch_size_per_predict):
        j = i + batch_size_per_predict
        batch_df["text"].iloc[
            i
        ] = labels  # each csv file includes the labels in the first row
        batch_df[i:j].to_csv(os.path.join(csv_folder_path, str(i) + batch_input_file))


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
