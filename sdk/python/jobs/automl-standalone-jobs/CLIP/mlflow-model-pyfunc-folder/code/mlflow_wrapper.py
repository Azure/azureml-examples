# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import mlflow
from PIL import Image
import pandas as pd
import torch
import tempfile

from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from common_constants import MLflowSchemaLiterals, MLflowLiterals, Tasks
from common_utils import create_temp_file, process_image, get_current_device
from typing import List, Tuple


class CLIPMLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for CLIP model."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Constructor for MLflow wrapper class
        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self._processor = None
        self._model = None
        self._task_type = task_type

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().
        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        if self._task_type == Tasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value:
            try:
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._processor = AutoProcessor.from_pretrained(model_dir)
                self._model = AutoModelForZeroShotImageClassification.from_pretrained(model_dir)
                self._device = get_current_device()
                self._model.to(self._device)

                print("Model loaded successfully")
            except Exception as e:
                print("Failed to load the the model.")
                print(e)
                raise
        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """Perform inference on the input data.
        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Input images for prediction and candidate labels.
        :type input_data: Pandas DataFrame with a first column name ["image"] of images where each
        image is in base64 String format, and second column name ["text"] where the first row contains the
        candidate labels and the remaining rows are ignored.
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns ["probs", "labels"]
        """
        # Decode the base64 image column
        decoded_images = input_data.loc[
            :, [MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
        ].apply(axis=1, func=process_image)

        try:
            captions = map(str.strip, input_data[MLflowSchemaLiterals.INPUT_COLUMN_TEXT].iloc[0].split(',')) # parse comma separated labels and remove trailing whitespace
            captions = list(filter(None, captions)) # remove any empty strings
            if len(captions) == 0:
                raise ValueError("No labels were provided")
        except Exception:
            raise ValueError(
                    "The provided labels cannot be parsed. The first row of the \"text\" column is expected to contain a string with the comma-separated labels"
            )

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            image_path_list = (
                decoded_images.iloc[:, 0]
                .map(lambda row: create_temp_file(row, tmp_output_dir))
                .tolist()
            )
            conf_scores = self.run_inference_batch(
                processor=self._processor,
                model=self._model,
                image_path_list=image_path_list,
                text_list=captions,
                task_type=self._task_type
            )

        df_result = pd.DataFrame(
            columns=[
                MLflowSchemaLiterals.OUTPUT_COLUMN_PROBS,
                MLflowSchemaLiterals.OUTPUT_COLUMN_LABELS,
            ]
        )

        labels = [captions] * len(conf_scores)
        df_result[MLflowSchemaLiterals.OUTPUT_COLUMN_PROBS], df_result[MLflowSchemaLiterals.OUTPUT_COLUMN_LABELS]\
            = (conf_scores.tolist(), labels)
        return df_result


    def run_inference_batch(
        self,
        processor,
        model,
        image_path_list: List,
        text_list: List,
        task_type: Tasks,
    ) -> Tuple[torch.tensor]:
        """Perform inference on batch of input images.
        :param test_args: Training arguments path.
        :type test_args: transformers.TrainingArguments
        :param image_processor: Preprocessing configuration loader.
        :type image_processor: transformers.AutoImageProcessor
        :param model: Pytorch model weights.
        :type model: transformers.AutoModelForImageClassification
        :param image_path_list: list of image paths for inferencing.
        :type image_path_list: List
        :param task_type: Task type of the model.
        :type task_type: Tasks
        :return: Predicted probabilities
        :rtype: Tuple of torch.tensor
        """

        image_list = [Image.open(img_path) for img_path in image_path_list]
        inputs = processor(text=text_list, images=image_list, return_tensors="pt", padding=True)
        inputs = inputs.to(self._device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        return probs