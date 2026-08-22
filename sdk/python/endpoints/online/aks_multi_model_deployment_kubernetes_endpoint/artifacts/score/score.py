import joblib
import json
import os
import pandas as pd
import logging
from pathlib import Path

# Declare models dictionary to hold the loaded models
models = None

# Initialize the models
def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the models in memory.
    """
    global models

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR")) / "models" # Path to the model directory
    logging.info(f"Model directory: {model_dir}")

    try:
        # Load both churn and segmentation models
        models = {
            "churn_model": joblib.load(model_dir / "churn.joblib"),         # For supervised classification
            "segmentation_model": joblib.load(model_dir / "segmentation.joblib")  # For unsupervised clustering
        }
        logging.info(f"Loaded models: {list(models.keys())}")
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}")
    except Exception as e:
        logging.error(f"Error during model loading in init: {e}")

# Run function to perform predictions
def run(raw_data):
    global models

    try:
        # Parse input data
        input_json = json.loads(raw_data)

        # Determine which model to run based on 'model_type' in input
        model_type = input_json.get("model_type", None)
        data = input_json.get("data", None)

        if model_type not in ["churn", "segmentation"]:
            raise ValueError("Invalid model_type. Choose either 'churn' or 'segmentation'.")

        if data is None:
            raise ValueError("Input data is missing.")

        input_data = pd.DataFrame(data)

        # Supervised classification: churn prediction
        if model_type == "churn":
            churn_model = models.get("churn_model")
            if churn_model is None:
                raise ValueError("Churn model not found.")
            
            # Perform churn prediction
            churn_predictions = churn_model.predict(input_data)
            result = {
                "churn_predictions": churn_predictions.tolist()
            }

        # Unsupervised clustering: segmentation
        elif model_type == "segmentation":
            segmentation_model = models.get("segmentation_model")
            if segmentation_model is None:
                raise ValueError("Segmentation model not found.")
            
            # Perform segmentation prediction
            segmentation_predictions = segmentation_model.predict(input_data)
            result = {
                "segmentation_predictions": segmentation_predictions.tolist()
            }

        return json.dumps(result)

    except Exception as e:
        error_message = str(e)
        logging.error(f"Error during prediction: {error_message}")
        return json.dumps({"error": error_message})
