# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import mlflow
from PIL import Image
import pandas as pd
import torch
import tempfile

from clip_mlflow_wrapper import CLIPMLFlowModelWrapper
from config import MLflowSchemaLiterals, Tasks
from typing import List, Tuple


class CLIPEmbeddingsMLFlowModelWrapper(CLIPMLFlowModelWrapper):
    """MLflow model wrapper for CLIP model, used for getting feature embeddings."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Initialize MLflow wrapper class.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__(task_type)
        self._supported_task = Tasks.EMBEDDINGS.value

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Input images and text for feature embeddings.
        :type input_data: Pandas DataFrame with a first column name ["image"] containing images where each
        row is an image in base64 String format or publicly accessible url format,
        and second column name ["text"] containing a string. The following cases are supported:
        - all rows in image column are populated with valid values and text column only contains empty string,
        - all rows in text column are populated with valid values and image column only contains empty string,
        - all rows in both columns are populated with valid values
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns "image_features" and/or "text_features"
        """
        # import in predict() since vision_utils.py is added during model export
        from vision_utils import create_temp_file, process_image_pandas_series
        has_images, has_text = CLIPEmbeddingsMLFlowModelWrapper.validate_input(input_data)

        if has_images:
            # Decode the base64 image column
            decoded_images = input_data.loc[
                :, [MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
            ].apply(axis=1, func=process_image_pandas_series)

        if has_text:
            text_list = input_data[MLflowSchemaLiterals.INPUT_COLUMN_TEXT].tolist()
        else:
            text_list = None

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            if has_images:
                image_path_list = (
                    decoded_images.iloc[:, 0]
                    .map(lambda row: create_temp_file(row, tmp_output_dir)[0])
                    .tolist()
                )
            else:
                image_path_list = None
            image_features, text_features = self.run_inference_batch(
                image_path_list=image_path_list,
                text_list=text_list,
            )

        df_result = pd.DataFrame()

        if image_features is not None:
            df_result[MLflowSchemaLiterals.OUTPUT_COLUMN_IMAGE_FEATURES] = image_features.tolist()
        if text_features is not None:
            df_result[MLflowSchemaLiterals.OUTPUT_COLUMN_TEXT_FEATURES] = text_features.tolist()
        return df_result

    def run_inference_batch(
        self,
        image_path_list: List,
        text_list: List,
    ) -> Tuple[torch.tensor]:
        """Perform inference on batch of input images.

        :type image_path_list: List[str]
        :param text_list: list of text strings for inferencing
        :type text_list: List[str]
        :return: image features and text features
        :rtype: Tuple where each value is either torch.tensor of size (#inputs, 512) or None
        """
        if image_path_list:
            image_list = [Image.open(img_path) for img_path in image_path_list]
            inputs = self._processor(text=None, images=image_list, return_tensors="pt", padding=True)
            inputs = inputs.to(self._device)
            image_features = self._model.get_image_features(**inputs)
        else:
            image_features = None

        if text_list:
            inputs = self._processor(text=text_list, images=None, return_tensors="pt", padding=True)
            inputs = inputs.to(self._device)
            text_features = self._model.get_text_features(**inputs)
        else:
            text_features = None

        return image_features, text_features

    @staticmethod
    def validate_input(input_data):
        """Validate input and raise exception if input is invalid.

        :param input_data: input to validate
        :type input_data: pandas.DataFrame
        """
        # Handle case where entire column is NaN, because batch inference
        # will read in empty column from CSV as NaN
        if input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE].isna().all():
            input_data.drop(columns=MLflowSchemaLiterals.INPUT_COLUMN_IMAGE, inplace=True)
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE] = ""

        if input_data[MLflowSchemaLiterals.INPUT_COLUMN_TEXT].isna().all():
            input_data.drop(columns=MLflowSchemaLiterals.INPUT_COLUMN_TEXT, inplace=True)
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_TEXT] = ""

        error_string = """Embeddings cannot be retrieved for the given input. The following cases are supported:
        - all rows in image column are populated with public urls or base64 encoded images and
          text column only contains empty string,
        - all rows in text column are populated with strings and image column only contains empty string,
        - all rows in both columns are populated with valid values"""
        # Validate Input
        input_data_all = input_data.all()
        input_data_any = input_data.any()

        if input_data_any[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE] and \
                not input_data_all[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]:
            raise ValueError(error_string)

        if input_data_any[MLflowSchemaLiterals.INPUT_COLUMN_TEXT] and \
                not input_data_all[MLflowSchemaLiterals.INPUT_COLUMN_TEXT]:
            raise ValueError(error_string)

        if not input_data_any[MLflowSchemaLiterals.INPUT_COLUMN_TEXT] and \
                not input_data_any[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]:
            raise ValueError(error_string)

        has_images = input_data_all[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
        has_text = input_data_all[MLflowSchemaLiterals.INPUT_COLUMN_TEXT]

        return has_images, has_text
