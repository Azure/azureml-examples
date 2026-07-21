import argparse
import base64
import json
import os
import shutil
import urllib.request
from zipfile import ZipFile


# Direct question to ask the model.
DIRECT_QUESTION = "What's in this image?"


def download_and_unzip(dataset_parent_dir: str) -> None:
    """Download image dataset and unzip it.

    :param dataset_parent_dir: dataset parent directory to which dataset will be downloaded
    :type dataset_parent_dir: str
    """
    # Create directory, if it does not exist
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # download data
    download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip"
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
    sample_image = os.path.join(dataset_dir, "milk_bottle", "99.jpg")

    request_json = {
        "input_data": {
            "columns": ["image", "prompt", "direct_question"],
            "index": [0],
            "data": [
                [
                    base64.encodebytes(read_image(sample_image)).decode("utf-8"),
                    "",
                    DIRECT_QUESTION,
                ]
            ],
        }
    }

    request_file_name = os.path.join(dataset_dir, "sample_request_data.json")

    with open(request_file_name, "w") as request_file:
        json.dump(request_json, request_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for text generation based on images and text."
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="Dataset location"
    )

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    dataset_dir = download_and_unzip(
        dataset_parent_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), args.data_path
        ),
    )

    prepare_data_for_online_inference(dataset_dir=dataset_dir)
