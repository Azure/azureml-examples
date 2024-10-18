import os
import base64
from PIL import Image
from tqdm import tqdm
import io
import numpy as np
import SimpleITK as sitk


def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


def read_dcm(dicom_path):
    # Read the DICOM file
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_path)
    dicom = reader.Execute()

    # Extract the pixel array
    img_array = sitk.GetArrayFromImage(dicom)[0, :, :]

    # Normalize the pixel values to the range [0, 255]
    img_array = img_array.astype(np.float32)
    img_array = (
        (img_array - np.min(img_array))
        / (np.max(img_array) - np.min(img_array))
        * 255.0
    )
    img_array = img_array.astype(np.uint8)

    # Convert the pixel array to a PIL Image
    image = Image.fromarray(img_array)

    # Save the image to a BytesIO object
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")

    image_bytes = image_bytes.getvalue()

    return image_bytes


def get_files_path(root_folder):
    print("--------Start Loading Image Files--------")
    files_data = {}
    for _, (folder, _, images) in enumerate(os.walk(root_folder)):
        for count, image_file in tqdm(enumerate(images), total=len(images)):
            file_path = os.path.join(folder, image_file)
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                file_content = base64.encodebytes(read_image(file_path)).decode("utf-8")
            elif file_path.lower().endswith(".dcm"):
                file_content = base64.b64encode(read_dcm(file_path)).decode("utf-8")
            else:
                continue  # Skip non-image files

            files_data[image_file] = {"file": file_content, "text": "", "index": count}
    return files_data


def get_text(texts):
    print("--------Start Loading Text--------")
    text_data = {}
    for count, text in tqdm(enumerate(texts), total=len(texts)):
        text_data[text] = {"file": "", "text": text, "index": count}
    return text_data


## Normalization Example Here
def normalize_volume(volume):
    volume[volume < -1000] = -1000
    volume[volume > 1000] = 1000
    normalized_volume = (volume + 1000) / 2000
    normalized_volume = (
        (normalized_volume - normalized_volume.min())
        * 255.0
        / (normalized_volume.max() - normalized_volume.min())
    )
    return normalized_volume


def convert_volume_to_slices(volume, output_dir, filename_prefix):
    for i in range(volume.shape[2]):
        slice_img = volume[:, :, i]
        slice_img = slice_img.astype(np.uint8)
        slice_img = (
            Image.fromarray(slice_img)
            .transpose(Image.ROTATE_90)
            .transpose(Image.FLIP_LEFT_RIGHT)
        )
        slice_img.save(os.path.join(output_dir, f"{filename_prefix}_slice{i}.png"))
