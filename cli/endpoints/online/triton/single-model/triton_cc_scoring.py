# from azureml_inference_server_http.prepost_server.prepost.user.model_handler_base import ModelHandlerBase
from PIL import Image
import numpy as np
import os
from pathlib import Path, PurePath
import io


class ModelHandlerBase:
    """
    ModelHandlerBase provides a template for user ModelHandler classes.

    The AML Prepost Server looks for a class called ModelHandler in your score.py file.
    ModelHandler must have preprocess and postprocess methods that each take two parameters-
    data and context.

    Your ModelHandler class does not need to inherit from ModelHandlerBase as long
    as preprocess(data, context) and postprocess(data, context) are provided.
    """

    def __init__(self):
        pass

    def preprocess(self, data, context):
        """
        Example preprocess method.

        Read raw bytes from the request, process the data, return KFServing input.

        Parameters:
            data (obj): the request body
            context (dict): HTTP context information

        Returns:
            dict: Json serializable KFServing input
        """
        print("[[[ModelHandlerBase]]] pre-processing")

    def postprocess(self, data, context):
        """
        Example postprocess method.

        Reshape output tensors, find the cor/ect class name for the prediction.

        Parameters:
            data (dict): The model server response
            context (dict): HTTP context information

        Returns:
            (string/bytes, string) Json serializable string or raw bytes, response content type
        """
        print("[[[ModelHandlerBase]]] post-processing")


class ModelHandler(ModelHandlerBase):
    def __init__(self):
        self.path, self.name, self.version = self.infer_model()
        self.label_dict = self.load_densenet_labels()
        self.c = 3
        self.h = 224
        self.w = 224

    def infer_model(self):
        model_dir = Path(os.getenv("AZUREML_MODEL_DIR")) / "models/triton"
        model = next(model_dir.glob("*/*/"))
        return (model, *PurePath(model).parts[-2:])

    def load_densenet_labels(self):
        label_path = Path(os.getenv("AML_APP_ROOT")) / "densenet_labels.txt"
        label_file = open(label_path, "r")
        labels = label_file.read().split("\n")
        label_dict = dict(enumerate(labels))

        return label_dict

    def preprocess(self, img_content, context):
        """Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        """
        bytz = io.BytesIO(img_content)

        img = Image.open(bytz)

        sample_img = img.convert("RGB")

        resized_img = sample_img.resize((self.w, self.h), Image.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        typed = resized.astype(np.float32)

        # scale for INCEPTION
        scaled = (typed / 128) - 1

        # Swap to CHW
        ordered = np.transpose(scaled, (2, 0, 1))

        # Channels are in RGB order. Currently model configuration data
        # doesn't provide any information as to other channel orderings
        # (like BGR) so we just assume RGB.

        img_array = np.array(ordered, dtype=np.float32)[None, ...]

        input = {
            "name": "data_0",
            "shape": img_array.shape,
            "datatype": "FP32",
            "data": img_array,
        }

        output = {"name": "fc6_1"}

        payload = {
            "model_name": self.name,
            "model_version": self.version,
            "inputs": [input],
            "outputs": [output],
        }

        return payload

    def postprocess(self, out_data, context):
        """
        Example postprocess method.

        Reshape output tensors, find the cor/ect class name for the prediction.

        Parameters:
            data (dict): The model server response
            context (dict): HTTP context information

        Returns:
            (string/bytes, string) Json serializable string or raw bytes, response content type
        """
        labels = [
            self.label_dict[np.argmax(output["data"])]
            for output in out_data["outputs"]
            if output["name"] == "fc6_1"
        ]
        return (str(labels), context)
