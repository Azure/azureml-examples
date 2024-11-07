import numpy as np
import json
import os
import ssl
import base64
import json
import matplotlib.pyplot as plt


def allowSelfSignedHttps(allowed):
    """Bypass the server certificate verification on client side"""
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(
    True
)  # this line is needed if you use self-signed certificate in your scoring service.


def read_image(image_path):
    """Read image pixel data from a file path.
    Return image pixel data as an array.
    """
    with open(image_path, "rb") as f:
        return f.read()


def decode_json_to_array(json_encoded):
    """Decode an image pixel data array in JSON.
    Return image pixel data as an array.
    """
    # Parse the JSON string
    array_metadata = json.loads(json_encoded)
    # Extract Base64 string, shape, and dtype
    base64_encoded = array_metadata["data"]
    shape = tuple(array_metadata["shape"])
    dtype = np.dtype(array_metadata["dtype"])
    # Decode Base64 to byte string
    array_bytes = base64.b64decode(base64_encoded)
    # Convert byte string back to NumPy array and reshape
    array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
    return array


def plot_segmentation_masks(original_image, segmentation_masks):
    """Plot a list of segmentation mask over an image."""
    fig, ax = plt.subplots(1, len(segmentation_masks) + 1, figsize=(10, 5))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")

    for i, mask in enumerate(segmentation_masks):
        ax[i + 1].imshow(original_image)
        ax[i + 1].set_title(f"Mask {i+1}")
        mask_temp = original_image.copy()
        mask_temp[mask > 128] = [255, 0, 0, 255]
        mask_temp[mask <= 128] = [0, 0, 0, 0]
        ax[i + 1].imshow(mask_temp, alpha=0.9)
    plt.show()
