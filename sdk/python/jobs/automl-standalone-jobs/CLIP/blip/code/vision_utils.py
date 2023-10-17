# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper utils for vision Mlflow models."""

import logging
import tempfile
import pandas as pd
import base64
import io
import re
import requests
import torch

from PIL import Image, UnidentifiedImageError
from typing import Tuple, Union


logger = logging.getLogger(__name__)

# Uncomment the following line for mlflow debug mode
# logging.getLogger("mlflow").setLevel(logging.DEBUG)


def create_temp_file(request_body: bytes, parent_dir: str) -> Tuple[str, Image.Image]:
    """Create temporory file from image bytes, save image and return path to the file and PIL Image.

    :param request_body: Image
    :type request_body: bytes
    :param parent_dir: directory name
    :type parent_dir: str
    :return: Path to the file, PIL Image
    :rtype: Tuple[str, Image.Image]
    """
    with tempfile.NamedTemporaryFile(dir=parent_dir, mode="wb", delete=False) as image_file_fp:
        img_path = image_file_fp.name + ".png"
        try:
            img = Image.open(io.BytesIO(request_body))
        except UnidentifiedImageError as e:
            logger.error("Invalid image format. Please use base64 encoding for input images.")
            raise e
        img.save(img_path)
        return img_path, img


def process_image(image: Union[str, bytes]) -> bytes:
    """Process image.

    If input image is in bytes format, return it as it is.
    If input image is in base64 string format, decode it to bytes.
    If input image is in url format, download it and return bytes.
    https://github.com/mlflow/mlflow/blob/master/examples/flower_classifier/image_pyfunc.py

    :param image: image in base64 string format or url or bytes.
    :type image: string or bytes
    :return: decoded image.
    :rtype: bytes
    """
    if isinstance(image, bytes):
        return image
    elif isinstance(image, str):
        if _is_valid_url(image):
            try:
                response = requests.get(image)
                response.raise_for_status()  # Raise exception in case of unsuccessful response code.
                image = response.content
                return image
            except requests.exceptions.RequestException as ex:
                raise ValueError(f"Unable to retrieve image from url string due to exception: {ex}")
        else:
            try:
                return base64.b64decode(image)
            except ValueError:
                raise ValueError("The provided image string cannot be decoded. "
                                 "Expected format is base64 string or url string.")
    else:
        raise ValueError(f"Image received in {type(image)} format which is not supported. "
                         "Expected format is bytes, base64 string or url string.")


def process_image_pandas_series(image_pandas_series: pd.Series) -> pd.Series:
    """Process image in Pandas series form.

    If input image is in bytes format, return it as it is.
    If input image is in base64 string format, decode it to bytes.
    If input image is in url format, download it and return bytes.
    https://github.com/mlflow/mlflow/blob/master/examples/flower_classifier/image_pyfunc.py

    :param img: pandas series with image in base64 string format or url or bytes.
    :type img: pd.Series
    :return: decoded image in pandas series format.
    :rtype: Pandas Series
    """
    image = image_pandas_series[0]
    return pd.Series(process_image(image))


def _is_valid_url(text: str) -> bool:
    """Check if text is url or base64 string.

    :param text: text to validate
    :type text: str
    :return: True if url else false
    :rtype: bool
    """
    regex = (
        "((http|https)://)(www.)?"
        + "[a-zA-Z0-9@:%._\\+~#?&//=\\-]"
        + "{2,256}\\.[a-z]"
        + "{2,6}\\b([-a-zA-Z0-9@:%"
        + "._\\+~#?&//=]*)"
    )
    p = re.compile(regex)

    # If the string is empty
    # return false
    if str is None:
        return False

    # Return if the string
    # matched the ReGex
    if re.search(p, text):
        return True
    else:
        return False


def get_current_device() -> torch.device:
    """Get current cuda device.

    :return: current device
    :rtype: torch.device
    """
    # check if GPU is available
    if torch.cuda.is_available():
        try:
            # get the current device index
            device_idx = torch.distributed.get_rank()
        except RuntimeError as ex:
            if 'Default process group has not been initialized'.lower() in str(ex).lower():
                device_idx = 0
            else:
                logger.error(str(ex))
                raise ex
        return torch.device(type="cuda", index=device_idx)
    else:
        return torch.device(type="cpu")
