import argparse
import base64
import json
import os
import urllib.request
import pandas as pd
from zipfile import ZipFile


def download_and_unzip(dataset_parent_dir: str) -> None:
    """Download image dataset and unzip it.

    :param dataset_parent_dir: dataset parent directory to which dataset will be downloaded
    :type dataset_parent_dir: str
    """

    # Create data folder if it doesnt exist.
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # Download data
    download_url = "https://automlresources-prod.azureedge.net/datasets/AirBnb.zip"

    # Extract current dataset name from dataset url
    dataset_name = os.path.split(download_url)[-1].split(".")[0]

    # Get the data zip file path
    data_file = os.path.join(dataset_parent_dir, f"{dataset_name}.zip")

    # Download the dataset
    urllib.request.urlretrieve(download_url, filename=data_file)

    # Get dataset path for later use
    dataset_dir = os.path.join(dataset_parent_dir, dataset_name)
    # Extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall(path=dataset_parent_dir)
        print("done")
    # Delete zip file
    os.remove(data_file)
    return dataset_dir


def read_dataframe_from_csv(csv_file_path: str, dataset_dir: str) -> pd.DataFrame:
    """
    Read csv file and return dataframe.

    :param csv_file_path: csv file path
    :type csv_file_path: str
    :param dataset_dir: dataset directory
    :type dataset_dir: str
    :return: dataframe
    :rtype: pd.DataFrame
    """

    def image_to_str(img_path) -> str:
        with open(os.path.join(dataset_dir, img_path), "rb") as f:
            encoded_image = base64.encodebytes(f.read()).decode("utf-8")
            return encoded_image

    # We can pass image either as azureml url on data asset or as a base64 encoded string.
    # Here, we will be passing base64 encoded string.

    image_column_name = "picture_url"
    df = pd.read_csv(csv_file_path, nrows=2)
    df[image_column_name] = df.apply(lambda x: image_to_str(x["picture_url"]), axis=1)

    return df


def prepare_data_for_batch_inference(dataset_dir: str) -> None:
    """Prepare image folder and csv file for batch inference.

    This function will move all images to a single image folder and also create a csv
    file with images in base64 format.
    :param dataset_dir: dataset directory
    :type dataset_dir: str
    """

    # Read sample rows from dataset
    # Initialize dataset specific fields
    csv_file_path = os.path.join(dataset_dir, "airbnb_multiclass_dataset.csv")
    # get dataframe and change image path to base64 encoded string
    df = read_dataframe_from_csv(csv_file_path=csv_file_path, dataset_dir=dataset_dir)

    csv_file_name = "multimodal_multiclass_classification_list.csv"
    # Directory in which we have our images
    df.to_csv(os.path.join(dataset_dir, csv_file_name), index=False, header=True)


def prepare_data_for_online_inference(dataset_dir: str) -> None:
    """Prepare request json for online inference.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    """

    # Read a sample row from dataset
    # Initialize dataset specific fields
    csv_file_path = os.path.join(dataset_dir, "airbnb_multiclass_dataset.csv")
    # get dataframe and change image path to base64 encoded string
    df = read_dataframe_from_csv(csv_file_path, dataset_dir)

    request_json = {
        "input_data": {
            "columns": df.columns.values.tolist(),
            "data": df.values.tolist(),
        }
    }

    # Create request json
    request_file_name = os.path.join(dataset_dir, "sample_request_data.json")
    with open(request_file_name, "w") as request_file:
        json.dump(request_json, request_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for multimodal multi-class classification"
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="Dataset location"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="batch",
        help="Prepare data for online or batch inference",
    )

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    dataset_dir = download_and_unzip(
        dataset_parent_dir=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.data_path
        ),
    )
    if args.mode == "batch":
        prepare_data_for_batch_inference(dataset_dir=dataset_dir)
    elif args.mode == "online":
        prepare_data_for_online_inference(dataset_dir=dataset_dir)
