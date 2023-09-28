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
        
        has_images, has_text = self.validate_input(input_data)

        if has_images:
            # Decode the base64 image column
            decoded_images = input_data.loc[
                :, [MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
            ].apply(axis=1, func=process_image)

        if has_text:
            text_list = input_data['text'].tolist()
        else:
            text_list = None

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            if has_images:
                image_path_list = (
                    decoded_images.iloc[:, 0]
                    .map(lambda row: create_temp_file(row, tmp_output_dir))
                    .tolist()
                )
            else:
                image_path_list = None
            image_features, text_features = self.run_inference_batch(
                processor=self._processor,
                model=self._model,
                image_path_list=image_path_list,
                text_list=text_list,
                task_type=self._task_type
            )

        df_result = pd.DataFrame(
            columns=[
                MLflowSchemaLiterals.OUTPUT_IMAGE_FEATURES,
                MLflowSchemaLiterals.OUTPUT_TEXT_FEATURES,
            ]
        )

        if image_features is not None:
            df_result[MLflowSchemaLiterals.OUTPUT_IMAGE_FEATURES] = image_features.tolist()
        if text_features is not None:
            df_result[MLflowSchemaLiterals.OUTPUT_TEXT_FEATURES] = text_features.tolist()
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

        if image_path_list:
            image_list = [Image.open(img_path) for img_path in image_path_list]
            inputs = processor(text=None, images=image_list, return_tensors="pt", padding=True)
            inputs = inputs.to(self._device)
            image_features = model.get_image_features(**inputs)
        else:
            image_features = None

        if text_list:
            inputs = processor(text=text_list, images=None, return_tensors="pt", padding=True)
            inputs = inputs.to(self._device)
            text_features = model.get_text_features(**inputs)
        else:
            text_features = None

        return image_features, text_features
    
    def validate_input(self, input_data):
        # Validate Input
        input_data_all = input_data.all()
        input_data_any = input_data.any()

        if input_data_any['image'] and not input_data_all['image']:
            print("image column has some images but not all")
            raise

        if input_data_any['text'] and not input_data_all['text']:
            print("text column has some text but not all")
            raise

        if not input_data_any['text'] and not input_data_any['image']:
            print("text and image columns are empty")
            raise

        has_images = input_data_all['image']
        has_text = input_data_all['text']

        return has_images, has_text