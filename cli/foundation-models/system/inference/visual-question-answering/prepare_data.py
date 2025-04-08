import argparse
import base64
import json
import os
import shutil
import urllib.request
import pandas as pd
from zipfile import ZipFile


def download_and_unzip(dataset_parent_dir: str) -> None:
    """Download image dataset and unzip it.

    :param dataset_parent_dir: dataset parent directory to which dataset will be downloaded
    :type dataset_parent_dir: str
    """
    # Create directory, if it does not exist
    os.makedirs(dataset_parent_dir, exist_ok=True)

    # download data
    download_url = "https://automlsamplenotebookdata-adcuc7f7bqhhh8a4.b02.azurefd.net/image-object-detection/odFridgeObjects.zip"
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
    sample_image_1 = os.path.join(dataset_dir, "images", "99.jpg")
    sample_image_2 = os.path.join(dataset_dir, "images", "1.jpg")

    request_json = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0, 1],
            "data": [
                [
                    base64.encodebytes(read_image(sample_image_1)).decode("utf-8"),
                    # For BLIP2 append "Answer:" to the below prompt
                    "Describe the beverage in the image?",
                ],
                [
                    base64.encodebytes(read_image(sample_image_2)).decode("utf-8"),
                    # For BLIP2 append "Answer:" to the below prompt
                    "What are the drinks on the table?",
                ],
            ],
        }
    }

    request_file_name = os.path.join(dataset_dir, "sample_request_data.json")

    with open(request_file_name, "w") as request_file:
        json.dump(request_json, request_file)


def prepare_data_for_batch_inference(dataset_dir: str) -> None:
    """Prepare image folder and csv file for batch inference.

    This function will create a folder of csv files with images in base64 format.
    :param dataset_dir: dataset directory
    :type dataset_dir: str
    """

    csv_folder_path = os.path.join(dataset_dir, "batch")
    os.makedirs(csv_folder_path, exist_ok=True)
    batch_input_file = "batch_input.csv"
    dataset_dir = os.path.join(dataset_dir, "images")

    # Take 20 images
    image_list = []
    for i in range(1, 21):
        image_path = os.path.join(dataset_dir, str(i) + ".jpg")
        data = read_image(image_path)
        data = base64.encodebytes(data).decode("utf-8")
        image_list.append(data)

    # Read questions list file
    with open("list_of_questions.txt", "r") as f:
        data = f.read()
        question_list = data.split("\n")
        # For BLIP2, append "Answer: " to the questions
        # question_list = [s + " Answer:" for s in question_list]

    # Divide the image & questions list into files of 10 rows each
    batch_size_per_predict = 10
    divided_image_list = [
        image_list[i * batch_size_per_predict : (i + 1) * batch_size_per_predict]
        for i in range(
            (len(image_list) + batch_size_per_predict - 1) // batch_size_per_predict
        )
    ]
    divided_question_list = [
        question_list[i * batch_size_per_predict : (i + 1) * batch_size_per_predict]
        for i in range(
            (len(question_list) + batch_size_per_predict - 1) // batch_size_per_predict
        )
    ]

    # Write to CSV files
    for l in range(0, len(divided_image_list)):
        dictionary = {"image": divided_image_list[l], "text": divided_question_list[l]}
        batch_df = pd.DataFrame(dictionary)
        filepath = os.path.join(csv_folder_path, str(l) + batch_input_file)
        batch_df.to_csv(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for visual-question-answering task"
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
