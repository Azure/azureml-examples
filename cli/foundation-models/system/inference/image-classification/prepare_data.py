import argparse
import base64
import json
import os
import shutil
import urllib.request
import pandas as pd
from zipfile import ZipFile


def download_and_unzip(dataset_parent_dir: str, is_multilabel_dataset: int) -> None:
    """Download image dataset and unzip it.

    :param dataset_parent_dir: dataset parent directory to which dataset will be downloaded
    :type dataset_parent_dir: str
    :param is_multilabel_dataset: flag to indicate if dataset is multi-label or not
    :type is_multilabel_dataset: int
    """
    # Create directory, if it does not exist
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # download data
    if is_multilabel_dataset == 0:
        download_url = "https://automlsamplenotebookdata-adcuc7f7bqhhh8a4.b02.azurefd.net/image-classification/fridgeObjects.zip"
    else:
        download_url = "https://automlsamplenotebookdata-adcuc7f7bqhhh8a4.b02.azurefd.net/image-classification/multilabelFridgeObjects.zip"
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


def prepare_data_for_online_inference(dataset_dir: str, is_multilabel: int = 0) -> None:
    """Prepare request json for online inference.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    :param is_multilabel: flag to indicate if dataset is multi-label or not
    :type is_multilabel: int
    """
    if is_multilabel == 0:
        sample_image = os.path.join(dataset_dir, "milk_bottle", "99.jpg")
    else:
        sample_image = os.path.join(dataset_dir, "images", "56.jpg")

    request_json = {
        "input_data": [base64.b64encode(read_image(sample_image)).decode("utf-8")],
    }

    request_file_name = os.path.join(dataset_dir, "sample_request_data.json")

    with open(request_file_name, "w") as request_file:
        json.dump(request_json, request_file)


def prepare_data_for_batch_inference(dataset_dir: str, is_multilabel: int = 0) -> None:
    """Prepare image folder and csv file for batch inference.

    This function will move all images to a single image folder and also create a csv
    file with images in base64 format.
    :param dataset_dir: dataset directory
    :type dataset_dir: str
    :param is_multilabel: flag to indicate if dataset is multi-label or not
    :type is_multilabel: int
    """
    image_list = []

    csv_file_name = (
        "image_classification_multilabel_lis.csv"
        if is_multilabel == 1
        else "image_classification_multiclass_list.csv"
    )

    for dir_name in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, dir_name)
        for path, _, files in os.walk(dir_path):
            for file in files:
                image = read_image(os.path.join(path, file))
                image_list.append(base64.b64encode(image).decode("utf-8"))
                shutil.move(os.path.join(path, file), dataset_dir)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        else:
            os.remove(dir_path)
    df = pd.DataFrame(image_list, columns=["image"]).sample(10)
    df.to_csv(
        os.path.join(os.path.dirname(dataset_dir), csv_file_name),
        index=False,
        header=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for image classification"
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="Dataset location"
    )
    parser.add_argument(
        "--is_multilabel", type=int, default=0, help="Is multilabel dataset"
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
            os.path.dirname(os.path.abspath(__file__)), args.data_path
        ),
        is_multilabel_dataset=args.is_multilabel,
    )

    if args.mode == "online":
        prepare_data_for_online_inference(
            dataset_dir=dataset_dir, is_multilabel=args.is_multilabel
        )
    else:
        prepare_data_for_batch_inference(
            dataset_dir=dataset_dir, is_multilabel=args.is_multilabel
        )
