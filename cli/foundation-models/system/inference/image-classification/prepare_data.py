import argparse
import base64
import json
import os
import urllib
from zipfile import ZipFile


def download_and_unzip(dataset_parent_dir: str, is_multilabel_dataset: int):

    # Create directory, if it does not exist
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # download data
    if is_multilabel_dataset == 0:
        download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip"
    else:
        download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip"
    print(f"Downloading data from {download_url}")

    # Extract current dataset name from dataset url
    dataset_name = os.path.basename(download_url).split(".")[0]

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


def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for image classification"
    )
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Dataset location"
    )
    parser.add_argument(
        "--is_multilabel", type=int, default=0, help="Is multilabel dataset"
    )

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    download_and_unzip(
        dataset_parent_dir=args.data_path,
        is_multilabel_dataset=args.is_multilabel,
    )

    if args.is_multilabel == 0:
        sample_image = os.path.join(args.data_path, "fridgeObjects", "milk_bottle", "99.jpg")
    else:
        sample_image = os.path.join(args.data_path, "multilabelFridgeObjects", "images", "56.jpg")

    request_json = {
        "inputs": {
            "image": [base64.encodebytes(read_image(sample_image)).decode("utf-8")],
        }
    }

    request_file_name = "sample_request_data.json"

    with open(request_file_name, "w") as request_file:
        json.dump(request_json, request_file)