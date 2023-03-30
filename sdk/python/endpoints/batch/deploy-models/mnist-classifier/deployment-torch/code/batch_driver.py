import os
import pandas as pd
import torch
import torchvision
import glob
from os.path import basename
from mnist_classifier import MnistClassifier
from typing import List


def init():
    global model
    global device

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder
    model_path = os.environ["AZUREML_MODEL_DIR"]
    model_file = glob.glob(f"{model_path}/*/*.pt")[-1]

    model = MnistClassifier()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(mini_batch: List[str]) -> pd.DataFrame:
    print(f"Executing run method over batch of {len(mini_batch)} files.")

    results = []
    with torch.no_grad():
        for image_path in mini_batch:
            image_data = torchvision.io.read_image(image_path).float()
            batch_data = image_data.expand(1, -1, -1, -1)
            input = batch_data.to(device)

            # perform inference
            predict_logits = model(input)

            # Compute probabilities, classes and labels
            predictions = torch.nn.Softmax(dim=-1)(predict_logits)
            predicted_prob, predicted_class = torch.max(predictions, axis=-1)

            results.append(
                {
                    "file": basename(image_path),
                    "class": predicted_class.numpy()[0],
                    "probability": predicted_prob.numpy()[0],
                }
            )

    return pd.DataFrame(results)
