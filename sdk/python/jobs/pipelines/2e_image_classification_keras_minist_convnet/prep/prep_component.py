# Converts MNIST-formatted files at the passed-in input path to training data output path and test data output path
import os
from pathlib import Path
from mldesigner import command_component, Input, Output


@command_component(
    name="prep_data",
    version="1",
    display_name="Prep Data",
    description="Convert data to CSV file, and split to training and test data",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    ),
)
def prepare_data_component(
    input_data: Input(type="uri_folder"),
    training_data: Output(type="uri_folder"),
    test_data: Output(type="uri_folder"),
):
    convert(
        os.path.join(input_data, "train-images-idx3-ubyte"),
        os.path.join(input_data, "train-labels-idx1-ubyte"),
        os.path.join(training_data, "mnist_train.csv"),
        60000,
    )
    convert(
        os.path.join(input_data, "t10k-images-idx3-ubyte"),
        os.path.join(input_data, "t10k-labels-idx1-ubyte"),
        os.path.join(test_data, "mnist_test.csv"),
        10000,
    )


def convert(img_file_path, label_file_path, out_file_path, n):
    img_file = open(img_file_path, "rb")
    label_file = open(label_file_path, "rb")
    output_file = open(out_file_path, "w")

    img_file.read(16)
    label_file.read(8)

    images = []
    for i in range(n):
        image = [ord(label_file.read(1))]
        for j in range(28 * 28):
            image.append(ord(img_file.read(1)))
        images.append(image)

    for image in images:
        output_file.write(",".join(str(pix) for pix in image) + "\n")

    img_file.close()
    label_file.close()
    output_file.close()
