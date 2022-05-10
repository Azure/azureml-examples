#from azureml_inference_server_http.prepost_server.prepost.user.model_handler_base import ModelHandlerBase 
from PIL import Image
import numpy as np
import os 
from pathlib import Path
import io
import logging

logger = logging.getLogger(__name__)

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

        Reshape output tensors, find the correct class name for the prediction.

        Parameters:
            data (dict): The model server response
            context (dict): HTTP context information

        Returns:
            (string/bytes, string) Json serializable string or raw bytes, response content type
        """
        print("[[[ModelHandlerBase]]] post-processing")


class ModelHandler(ModelHandlerBase):

    def __init__(self):
        self.label_dict = self.load_densenet_labels()
        self.dim = (3, 224, 224)

    def load_densenet_labels(self):
        folder_path = Path(os.getenv("AML_APP_ROOT")) 
        label_path = folder_path / "densenet_labels.txt"
        labels = open(label_path, "r").read().split("\n")
        return dict(enumerate(labels))

    def preprocess(self, img_content, context):
        """Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        """
        logger.info(context)
        #logger.info(img_content)
        #logger.info(type(img_content))
        #print(type(img_content))
        bytz = io.BytesIO(img_content)
        # logger.info(bytz) 
        #bytz.seek(15,0)

        img = Image.open(bytz)

        #logger.info(img)

        sample_img = img.convert("RGB")

        resized_img = sample_img.resize(self.dim[1:], Image.BILINEAR)
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

        logger.info(img_array.shape)

        return {"inputs" :img_array}

    def postprocess(self, max_label, context):
        """Post-process results to show the predicted label."""
        # asdfasdffds 

        final_label = self.label_dict[max_label]
        # return #f"{max_label} : {final_label}" 
