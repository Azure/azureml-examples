import os
import io
import sys
import json
import argparse
import traceback

import base64
from PIL import Image
import mlflow
import torch
import torchvision

_MODEL = None
_TRANSFORMS = None


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global _MODEL, _TRANSFORMS

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"))

    _MODEL = mlflow.pytorch.load_model(model_path)

    _TRANSFORMS = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    print("Init complete")


def request_to_model_input(request_body, transforms):
    """
    Expects a request of format { 'rows' : [ { 'image' : BASE64STRING } ] }
    Loads a base64 encoded image content into a numpy array for model to predict.
    Applies all the transforms necessary.
    """
    # verify request body
    assert isinstance(request_body, str), "request body has to be a str"
    request_json = json.loads(request_body)
    assert isinstance(request_json, dict), "request json has to be a dict"
    assert "rows" in request_json, "request json has to contain key 'rows'"
    assert isinstance(
        request_json["rows"], list
    ), "request json key 'rows' has to be a list"

    images = []
    for idx, row in enumerate(request_json["rows"]):
        try:
            input_image_base64 = row["image"]

            # data contains a base64 encode of a jpg image
            pil_image = Image.open(io.BytesIO(base64.b64decode(input_image_base64)))

            # https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
            # check pil_loader()
            pil_image = pil_image.convert("RGB")

            images.append(pil_image)
        except:
            raise Exception(
                "row on index {} could not be processed into a model input.\n{}".format(
                    idx, traceback.format_exc()
                )
            )

    return images


def run(request_body):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    global _MODEL, _TRANSFORMS

    try:
        # processes the http request body to obtain model input data
        input_images = request_to_model_input(request_body, _TRANSFORMS)

        # apply transforms
        input_data = []
        for image in input_images:
            input_data.append(_TRANSFORMS(image))

        input_data = torch.stack(input_data)

        # predict using the model
        model_output = _MODEL(input_data).clone().detach().numpy()

        # transform model output to request answer (json serializable)
        return [
            [
                {"prob": float(p_c), "class": int(i_c), "label": i_c}
                for i_c, p_c in enumerate(y_i)
            ]
            for i, y_i in enumerate(model_output)
        ]

    except BaseException as e:  # NOTE: this is a catch all
        return {"exception": traceback.format_exc()}


### FOR DEBUG/LOCAL USE ONLY ###


def main(cli_args=None):
    # this main function should be called only
    # when running this script locally from shell

    # create arguments
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(f"Test model")
    group.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to model folder",
    )
    group.add_argument(
        "--test_image",
        type=str,
        required=True,
        help="Path to model folder",
    )

    # runs on cli arguments
    args = parser.parse_args(cli_args)  # if None, runs on sys.argv

    # Load the image and transform as a payload compatible
    # with the run() method
    print("* Testing service with image:\n{}".format(args.test_image))
    with open(args.test_image, mode="rb") as in_file:
        image_64_encode = base64.b64encode(in_file.read()).decode("ascii")

    request_payload = {"rows": [{"image": image_64_encode}]}
    request_body = json.dumps(request_payload)

    # this variable will be used when script is served in Azure ML
    os.environ["AZUREML_MODEL_DIR"] = args.model_dir

    # call init() method like AzureML would
    init()

    # call run() method with fake request body
    response_json = run(request_body)

    print("* Test request call returned:")
    print(json.dumps(response_json, indent="    "))


if __name__ == "__main__":
    main()
