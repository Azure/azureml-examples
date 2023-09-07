# Read base64 encoded image from txt file and convert it to image file
import argparse
import re
import json
import io
import base64
from PIL import Image


def base64_str_to_image(response_file: str) -> bytes:
    with open(response_file) as f:
        serialized_image_json = f.read().strip()

    serialized_image_json=serialized_image_json.replace("\\\"","\"")[1:-1] 

    json_obj = json.loads(serialized_image_json)
    for obj in json_obj:
        text_prompt = obj["prompt"].strip()
        generated_image = obj["image"]
        img = Image.open(io.BytesIO(base64.b64decode(generated_image)))
        text_prompt=re.sub(r"[^a-zA-Z0-9 ]+", "", text_prompt)
        img.save(text_prompt + ".jpg", "JPEG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for image classification")
    parser.add_argument("--response_file", type=str, default="generated_image.txt", help="File having image response from endpoint.")
    args, unknown = parser.parse_known_args()
    
    base64_str_to_image(args.response_file)
