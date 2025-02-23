import gradio as gr
from io import BytesIO
import json
from PIL import Image
import requests
import base64
import io

AZURE_ENDPOINT = "AZURE_AI_MAAS_ENDPOINT" + "/images/generations"
KEY = "AZURE_AI_MAAS_ENDPOINT_KEY"


def save_and_generate_image(
    input_image, prompt, output_format, strength, negative_prompt, seed
):
    print(f"Image Prompt is : {prompt}")

    # Convert the inital image object to bytes
    buffered = io.BytesIO()
    input_image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # Encode the bytes to base64
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")

    image = generate_image(
        prompt, output_format, negative_prompt, strength, encoded_string, seed
    )
    image_path = "./generated_image.png"
    image.save(image_path)
    print(f"Image saved to {image_path}")

    return image_path


def generate_image(
    prompt, output_format, negative_prompt, strength, encoded_string, seed
):

    params = {
        "prompt": prompt,
        "image_prompt": {"image": encoded_string},
        "output_format": output_format,
        "seed": seed,
    }

    if negative_prompt:
        params["negative_prompt"] = negative_prompt

    if strength:
        params["image_prompt"]["strength"] = strength

    headers = {"Authorization": f"{KEY}", "Accept": "application/json"}

    response = requests.post(AZURE_ENDPOINT, headers=headers, json=params)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTPError: {e}")
        print(f"Response content: {response.content}")
        raise

    # Decode response
    image_data = base64.b64decode(response.json()["image"])
    output_image = Image.open(BytesIO(image_data))

    return output_image


demo = gr.Interface(
    fn=save_and_generate_image,
    inputs=[
        gr.Image(type="pil", label="Initial Image"),
        gr.Textbox(
            label="Enter your Image Prompt", placeholder="Describe your image..."
        ),
        gr.Radio(choices=["jpeg", "png"], label="Output Format", value="jpeg"),
        gr.Slider(
            minimum=0, maximum=1, step=0.01, label="Strength (optional)", value=0.5
        ),
        gr.Textbox(
            label="Negative Prompt (optional)", placeholder="What to avoid in the image"
        ),
        gr.Slider(minimum=0, maximum=1000, step=1, label="Seed (optional)", value=0),
    ],
    outputs=[gr.Image(label="Generated Image")],
    title="Stability AI on Azure AI | Image to Image",
)

if __name__ == "__main__":
    demo.launch()
