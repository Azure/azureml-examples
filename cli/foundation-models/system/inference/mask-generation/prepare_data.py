import argparse
import base64
import json
import os
import shutil
import urllib.request
import pandas as pd
from zipfile import ZipFile

def download_from_blob(url, dest):
    """Download from Azure Blob Storage with az cli fallback for auth."""
    try:
        urllib.request.urlretrieve(url, filename=dest)
    except urllib.error.HTTPError as e:
        import subprocess
        from urllib.parse import urlparse

        print(f"Anonymous download failed ({e}), trying az cli...")
        parsed = urlparse(url)
        account_name = parsed.hostname.split(".")[0]
        parts = parsed.path.lstrip("/").split("/", 1)
        container_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        cmd = [
            "az", "storage", "blob", "download",
            "--account-name", account_name,
            "--container-name", container_name,
            "--name", blob_name,
            "--file", dest,
            "--auth-mode", "login",
            "--only-show-errors",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"az cli auth-mode login failed: {result.stderr}")
            # Try with account key as last resort
            cmd_key = [
                "az", "storage", "blob", "download",
                "--account-name", account_name,
                "--container-name", container_name,
                "--name", blob_name,
                "--file", dest,
                "--only-show-errors",
            ]
            result2 = subprocess.run(cmd_key, capture_output=True, text=True)
            if result2.returncode != 0:
                print(f"az cli key-based also failed: {result2.stderr}")
                raise RuntimeError(
                    f"Cannot download {url}. Both anonymous and authenticated "
                    f"downloads failed. The storage account may have network "
                    f"restrictions blocking this runner."
                )
        print("az cli download succeeded.")




def download_and_unzip(dataset_parent_dir: str) -> None:
    """Download image dataset and unzip it.

    :param dataset_parent_dir: dataset parent directory to which dataset will be downloaded
    :type dataset_parent_dir: str
    """
    # Create directory, if it does not exist
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # download data

    download_url = "https://automlsamplenotebookdata.blob.core.windows.net/image-object-detection/odFridgeObjects.zip"
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
    download_from_blob(download_url, data_file)

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


def prepare_data_for_batch_inference(dataset_dir: str) -> None:
    """Prepare csv file for batch inference.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    """
    csv_folder_path = os.path.join(dataset_dir, "batch")
    os.makedirs(csv_folder_path, exist_ok=True)
    csv_file_name = "sample_request_data.csv"

    sample_image = os.path.join(dataset_dir, "images", "99.jpg")

    # Convert the image to base64 and prepare the data for DataFrame
    image_base64 = base64.encodebytes(read_image(sample_image)).decode("utf-8")
    data = [
        [image_base64, "[[[280,320]], [[300,350]]]", "", "", False],
        [image_base64, "[[[280,320], [300,350]]]", "", "", False],
        [image_base64, "", "[[125,240,375,425]]", "", False],
        [image_base64, "[[[280,320]]]", "[[125,240,375,425]]", "", False],
        [image_base64, "[[[280,320]]]", "[[125,240,375,425]]", "[[0]]", False],
    ]

    # Create DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "image",
            "input_points",
            "input_boxes",
            "input_labels",
            "multimask_output",
        ],
    )

    # Save DataFrame to CSV
    df.to_csv(os.path.join(dataset_dir, csv_folder_path, csv_file_name), index=False)


def prepare_data_for_online_inference(dataset_dir: str) -> None:
    """Prepare request json for online inference.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    """
    sample_image = os.path.join(dataset_dir, "images", "99.jpg")
    request_json = {
        "input_data": {
            "columns": [
                "image",
                "input_points",
                "input_boxes",
                "input_labels",
                "multimask_output",
            ],
            "index": [0, 1, 2, 3, 4],
            "data": [
                # segmentation mask per input point
                [
                    base64.encodebytes(read_image(sample_image)).decode("utf-8"),
                    "[[[280,320]], [[300,350]]]",
                    "",
                    "",
                    True,
                ],
                # single segmentation mask for multiple input points
                [
                    base64.encodebytes(read_image(sample_image)).decode("utf-8"),
                    "[[[280,320], [300,350]]]",
                    "",
                    "",
                    True,
                ],
                # single segmentation mask per single bounding box
                [
                    base64.encodebytes(read_image(sample_image)).decode("utf-8"),
                    "",
                    "[[125,240,375,425]]",
                    "",
                    True,
                ],
                # segmentation mask using both bounding box and input points
                [
                    base64.encodebytes(read_image(sample_image)).decode("utf-8"),
                    "[[[280,320]]]",
                    "[[125,240,375,425]]",
                    "",
                    True,
                ],
                # segmentation mask using both bounding box and input points and labels
                [
                    base64.encodebytes(read_image(sample_image)).decode("utf-8"),
                    "[[[280,320]]]",
                    "[[125,240,375,425]]",
                    "[[0]]",
                    True,
                ],
            ],
        }
    }

    request_file_name = os.path.join(dataset_dir, "sample_request_data.json")

    with open(request_file_name, "w") as request_file:
        json.dump(request_json, request_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for image object detection"
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="Dataset location"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
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
