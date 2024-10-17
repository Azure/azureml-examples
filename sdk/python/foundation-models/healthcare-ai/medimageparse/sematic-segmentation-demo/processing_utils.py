import numpy as np
import matplotlib.pyplot as plt
import pydicom
import nibabel as nib
import SimpleITK as sitk
from io import BytesIO
from PIL import Image
from skimage import transform, measure
import urllib.request
import json
import base64
import cv2


"""
    This script contains utility functions for reading and processing different imaging modalities.
"""
CT_WINDOWS = {
    "abdomen": [-150, 250],
    "lung": [-1000, 1000],
    "pelvis": [-55, 200],
    "liver": [-25, 230],
    "colon": [-68, 187],
    "pancreas": [-100, 200],
}


def process_intensity_image(image_data, is_CT, site=None):
    # process intensity-based image. If CT, apply site specific windowing

    # image_data: 2D numpy array of shape (H, W)

    # return: 3-channel numpy array of shape (H, W, 3) as model input

    if is_CT:
        # process image with windowing
        if site and site in CT_WINDOWS:
            window = CT_WINDOWS[site]
        else:
            raise ValueError(f"Please choose CT site from {CT_WINDOWS.keys()}")
        lower_bound, upper_bound = window
    else:
        # process image with intensity range 0.5-99.5 percentile
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)

    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - image_data_pre.min())
        / (image_data_pre.max() - image_data_pre.min())
        * 255.0
    )

    # pad to square with equal padding on both sides
    shape = image_data_pre.shape
    if shape[0] > shape[1]:
        pad = (shape[0] - shape[1]) // 2
        pad_width = ((0, 0), (pad, pad))
    elif shape[0] < shape[1]:
        pad = (shape[1] - shape[0]) // 2
        pad_width = ((pad, pad), (0, 0))
    else:
        pad_width = None

    if pad_width is not None:
        image_data_pre = np.pad(
            image_data_pre, pad_width, "constant", constant_values=0
        )

    # Important: resize image to 1024x1024
    image_size = 1024
    resize_image = transform.resize(
        image_data_pre,
        (image_size, image_size),
        order=3,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    )

    # convert to 3-channel image
    resize_image = np.stack([resize_image] * 3, axis=-1)

    return resize_image.astype(np.uint8)


def read_dicom(image_path, is_CT, site=None):
    # read dicom file and return pixel data

    # dicom_file: str, path to dicom file
    # is_CT: bool, whether image is CT or not
    # site: str, one of CT_WINDOWS.keys()
    # return: 2D numpy array of shape (H, W)

    ds = pydicom.dcmread(image_path)
    image_array = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept

    image_array = process_intensity_image(image_array, is_CT, site)
    # Step 1: Convert NumPy array to an image
    image = Image.fromarray(image_array)

    # Step 2: Save image to a BytesIO buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # or "JPEG", depending on your preference
    buffer.seek(0)

    return buffer


def read_nifti(
    image_path, is_CT, slice_idx, site=None, HW_index=(0, 1), channel_idx=None
):
    # read nifti file and return pixel data

    # image_path: str, path to nifti file
    # is_CT: bool, whether image is CT or not
    # slice_idx: int, slice index to read
    # site: str, one of CT_WINDOWS.keys()
    # HW_index: tuple, index of height and width in the image shape
    # return: 2D numpy array of shape (H, W)

    nii = nib.load(image_path)
    image_array = nii.get_fdata()

    if HW_index != (0, 1):
        image_array = np.moveaxis(image_array, HW_index, (0, 1))

    # get slice
    if channel_idx is None:
        image_array = image_array[:, :, slice_idx]
    else:
        image_array = image_array[:, :, slice_idx, channel_idx]

    image_array = process_intensity_image(image_array, is_CT, site)
    # Step 1: Convert NumPy array to an image
    image = Image.fromarray(image_array)

    # Step 2: Save image to a BytesIO buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # or "JPEG", depending on your preference
    buffer.seek(0)

    return buffer


def read_rgb(image_path):
    # read RGB image and return resized pixel data

    # image_path: str, path to RGB image
    # return: BytesIO buffer

    # read image into numpy array
    image = Image.open(image_path)
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    # pad to square with equal padding on both sides
    shape = image.shape
    if shape[0] > shape[1]:
        pad = (shape[0] - shape[1]) // 2
        pad_width = ((0, 0), (pad, pad), (0, 0))
    elif shape[0] < shape[1]:
        pad = (shape[1] - shape[0]) // 2
        pad_width = ((pad, pad), (0, 0), (0, 0))
    else:
        pad_width = None

    if pad_width is not None:
        image = np.pad(image, pad_width, "constant", constant_values=0)

    # resize image to 1024x1024 for each channel
    image_size = 1024
    resize_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for i in range(3):
        resize_image[:, :, i] = transform.resize(
            image[:, :, i],
            (image_size, image_size),
            order=3,
            mode="constant",
            preserve_range=True,
            anti_aliasing=True,
        )

    # Step 1: Convert NumPy array to an image
    resize_image = Image.fromarray(resize_image)

    # Step 2: Save image to a BytesIO buffer
    buffer = BytesIO()
    resize_image.save(buffer, format="PNG")  # or "JPEG", depending on your preference
    buffer.seek(0)

    return buffer


def get_instances(mask):
    # get intances from binary mask
    seg = sitk.GetImageFromArray(mask)
    filled = sitk.BinaryFillhole(seg)
    d = sitk.SignedMaurerDistanceMap(
        filled, insideIsPositive=False, squaredDistance=False, useImageSpacing=False
    )

    ws = sitk.MorphologicalWatershed(d, markWatershedLine=False, level=1)
    ws = sitk.Mask(ws, sitk.Cast(seg, ws.GetPixelID()))
    ins_mask = sitk.GetArrayFromImage(ws)

    # filter out instances with small area outliers
    props = measure.regionprops_table(ins_mask, properties=("label", "area"))
    mean_area = np.mean(props["area"])
    std_area = np.std(props["area"])

    threshold = mean_area - 2 * std_area - 1
    ins_mask_filtered = ins_mask.copy()
    for i, area in zip(props["label"], props["area"]):
        if area < threshold:
            ins_mask_filtered[ins_mask == i] = 0

    return ins_mask_filtered


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


def plot_segmentation_masks(
    original_image, segmentation_masks, text_prompt=None, aspect_ratio="auto"
):
    """
    Plot a list of segmentation masks over an image with a controllable aspect ratio.

    Parameters:
    - original_image: numpy array
        The original image to be displayed as the background.
        It should be a 2D (grayscale) or 3D (RGB) array.
    - segmentation_masks: list of numpy arrays
        A list where each element is a segmentation mask corresponding to the original image.
        Each mask should be a 2D array with the same spatial dimensions as the original image.
    - text_prompt: string, optional
        A string containing mask names separated by '&'.
        If provided, these names will be used as titles for the masks.
        Example: 'Cell Nuclei & Cytoplasm & Background'
    - aspect_ratio: float or string, optional
        The aspect ratio for each subplot. Can be a numeric value, 'auto', or 'equal'.
        - If 'equal', each subplot will have equal aspect ratio (no distortion).
        - If 'auto' (default), the aspect ratio is determined automatically.
        - If a numeric value is provided, it sets the aspect ratio as y/x.
          For example, aspect_ratio=1 makes y-axis equal to x-axis.

    The function displays the original image alongside each segmentation mask overlaid in red.
    """
    # Ensure the image has at least 3 channels (RGB)
    if original_image.ndim == 2:
        # Convert grayscale to RGB by stacking the 2D array into three channels
        original_image = np.stack((original_image,) * 3, axis=-1)
    elif original_image.shape[2] > 3:
        # If more than 3 channels, take the first three
        original_image = original_image[:, :, :3]

    num_masks = len(segmentation_masks)

    # Create subplots: one for the original image and one for each mask
    fig, ax = plt.subplots(1, num_masks + 1, figsize=(5 * (num_masks + 1), 5))

    # If there's only one subplot, wrap it in a list for consistency
    if num_masks == 0:
        ax = [ax]
    elif num_masks == 1:
        ax = [ax[0], ax[1]]

    # Display the original image
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].set_aspect(aspect_ratio)

    # Remove axes for all subplots
    for a in ax:
        a.axis("off")

    # Generate mask names
    if text_prompt:
        # Split the text prompt into mask names
        mask_names = [name.strip() for name in text_prompt.split("&")]
        # Check if the number of names matches the number of masks
        if len(mask_names) != num_masks:
            print(
                "Warning: Number of mask names does not match number of masks. Using default names."
            )
            mask_names = [f"Mask {i+1}" for i in range(num_masks)]
    else:
        # Default mask names if no text prompt is provided
        mask_names = [f"Mask {i+1}" for i in range(num_masks)]

    # Overlay each mask on the original image
    for i, mask in enumerate(segmentation_masks):
        # Set the title for the subplot
        ax[i + 1].set_title(mask_names[i])

        # Create an overlay with the same dimensions as the original image
        overlay = np.zeros_like(original_image, dtype=np.uint8)

        # Define the mask threshold (assumes masks are in the range [0, 255])
        threshold = 128

        # Set the red channel where the mask is greater than the threshold
        overlay[mask > threshold, 0] = 255  # Red channel

        # Display the original image
        ax[i + 1].imshow(original_image)

        # Overlay the mask with transparency
        ax[i + 1].imshow(overlay, alpha=0.5)

        # Set the aspect ratio for each subplot
        ax[i + 1].set_aspect(aspect_ratio)

    plt.tight_layout()
    plt.show()


# Combined inference function to handle both NIFTI and RGB inputs
def run_inference(
    inference_config,
    file_path,
    text_prompt,
    is_CT=False,
    slice_idx=None,
    site=None,
    HW_index=(0, 1),
    channel_idx=None,
):
    """
    Runs inference on the provided image and text input using the specified configuration.

    Parameters:
    - inference_config: dict with endpoint URL, API key, and model deployment info.
    - file_path: str, path to the image file.
    - text_prompt: str, text prompt for the model input.
    - is_CT: bool, True if the image is a CT scan (only used for NIFTI).
    - slice_idx: int, slice index for NIFTI images.
    - site: Optional, additional parameter for NIFTI images.
    - HW_index: tuple, used for indexing height and width of NIFTI images.
    - channel_idx: Optional, channel index for NIFTI images.

    Returns:
    - sample_image_arr: np.ndarray, the original image as an array.
    - image_features: np.ndarray, the decoded image features.
    """

    # Get file extension from file_path
    if file_path.lower().endswith(".nii.gz"):
        file_extension = "nii.gz"
    else:
        file_extension = file_path.split(".")[-1].lower()

    # Read and encode image based on type
    if file_extension == "nii" or file_extension == "nii.gz":
        image_data = base64.encodebytes(
            read_nifti(
                file_path,
                is_CT,
                slice_idx,
                site=site,
                HW_index=HW_index,
                channel_idx=channel_idx,
            ).read()
        ).decode("utf-8")
        sample_image_arr = np.array(
            Image.open(
                read_nifti(
                    file_path,
                    is_CT,
                    slice_idx,
                    site=site,
                    HW_index=HW_index,
                    channel_idx=channel_idx,
                )
            )
        )
    elif file_extension == "png" or file_extension == "jpg" or file_extension == "jpeg":
        image_data = base64.encodebytes(read_rgb(file_path).read()).decode("utf-8")
        sample_image_arr = np.array(Image.open(read_rgb(file_path)))
    elif file_extension == "dcm":
        image_data = base64.encodebytes(
            read_dicom(file_path, is_CT, site=site).read()
        ).decode("utf-8")
        sample_image_arr = np.array(Image.open(read_dicom(file_path, is_CT, site=site)))
    else:
        raise ValueError("Unsupported image type. Use 'RGB' or 'NIFTI'.")

    # Prepare the request payload
    data = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0],
            "data": [[image_data, text_prompt]],
        }
    }
    body = str.encode(json.dumps(data))
    url = f"{inference_config['endpoint']}/score"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {inference_config['api_key']}",
    }
    deployment = inference_config.get("azureml_model_deployment", None)
    if deployment:
        headers["azureml-model-deployment"] = deployment

    # Send the request and handle response
    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        result_list = json.loads(result)

        # Decode image features from response
        image_features_str = result_list[0]["image_features"]
        text_features = result_list[0]["text_features"]
        image_features = decode_json_to_array(image_features_str)

        # Plot the segmentation masks over the original image
        plot_segmentation_masks(sample_image_arr, image_features, text_prompt)
    except urllib.error.HTTPError as error:
        print(f"The request failed with status code: {error.code}")
        print(error.info())
        print(error.read().decode("utf8", "ignore"))

    return sample_image_arr, image_features, text_features


from processing_utils import get_instances
import cv2


def plot_instance_segmentation_masks(
    original_image, segmentation_masks, text_prompt=None
):
    """Plot a list of segmentation mask over an image."""
    original_image = original_image[:, :, :3]
    fig, ax = plt.subplots(1, len(segmentation_masks) + 1, figsize=(10, 5))
    ax[0].imshow(original_image, cmap="gray")
    ax[0].set_title("Original Image")
    # grid off
    for a in ax:
        a.axis("off")

    instance_masks = [get_instances(1 * (mask > 127)) for mask in segmentation_masks]

    mask_names = [f"Mask {i+1}" for i in range(len(segmentation_masks))]
    if text_prompt:
        mask_names = text_prompt.split("&")
        for i in range(len(mask_names)):
            mask_names[i] = mask_names[i].strip()

    for i, mask in enumerate(instance_masks):
        ins_ids = np.unique(mask)
        count = len(ins_ids[ins_ids > 0])

        ax[i + 1].set_title(f"{mask_names[i]} ({count})")
        mask_temp = np.zeros_like(original_image)
        for ins_id in ins_ids:
            if ins_id == 0:
                continue
            mask_temp[mask == ins_id] = np.random.randint(0, 255, 3)
            if ins_id == 1:
                mask_temp[mask == ins_id] = [255, 0, 0]

        ax[i + 1].imshow(mask_temp, alpha=1)
        ax[i + 1].imshow(original_image, cmap="gray", alpha=0.5)

    plt.show()
