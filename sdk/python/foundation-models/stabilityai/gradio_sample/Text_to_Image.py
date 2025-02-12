import gradio as gr
from io import BytesIO
import json
from PIL import Image
import requests
import base64

AZURE_ENDPOINT = "AZURE_AI_MAAS_ENDPOINT" + "/images/generations"
KEY = "AZURE_AI_MAAS_ENDPOINT_KEY"


def save_and_generate_image(
    prompt,
    output_format,
    negative_prompt,
    seed,
    size,
    progress=gr.Progress(track_tqdm=True),
):
    print(f"Image Prompt is : {prompt}")

    image = generate_image(prompt, output_format, negative_prompt, seed, size)
    image_path = "./generated_image.png"
    image.save(image_path)
    print(f"Image saved to {image_path}")

    return image_path


def generate_image(prompt, output_format, negative_prompt, seed, size):

    params = {
        "prompt": prompt,
        "output_format": output_format,
        "size": size,
        "seed": seed,
    }

    if negative_prompt:
        params["negative_prompt"] = negative_prompt

    print(f"Sending request with params: {json.dumps(params, indent=2)}")

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
        gr.Textbox(
            label="Enter your Image Prompt", placeholder="Describe your image..."
        ),
        gr.Radio(choices=["jpeg", "png"], label="Output Format", value="jpeg"),
        gr.Textbox(
            label="Negative Prompt (optional)", placeholder="What to avoid in the image"
        ),
        gr.Slider(minimum=0, maximum=1000, step=1, label="Seed (optional)", value=0),
        gr.Radio(
            choices=[
                "672x1566",
                "768x1366",
                "836x1254",
                "916x1145",
                "1024x1024",
                "1145x916",
                "1254x836",
                "1366x768",
                "1566x672",
            ],
            label="Image Size",
            value="1024x1024",
        ),
    ],
    outputs=[gr.Image(label="Generated Image")],
    title="Stability AI on Azure AI | Text to Image",
)

if __name__ == "__main__":
    demo.launch()
