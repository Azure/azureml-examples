import mlflow
from PIL import Image
import pandas as pd
import torch
import tempfile
from transformers import CLIPProcessor, CLIPModel
from common_constants import (HFMiscellaneousLiterals, Tasks, MLFlowSchemaLiterals)
from common_utils import create_temp_file, process_image
from typing import List, Dict, Any, Tuple

class CLIPMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for stable diffusion models."""

    def __init__(
            self,
            task_type: str,
    ) -> None:
        """Constructor for MLflow wrapper class
        :param task_type: Task type used in training.
        :type task_type: str
        :param model_id: Hugging face model id corresponding to stable diffusion models supported by AML.
        :type model_id: str
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
        if self._task_type == Tasks.HF_MULTI_CLASS_IMAGE_CLASSIFICATION:
            try:
                # _map_location = "cuda" if torch.cuda.is_available() else "cpu"
                model_dir = context.artifacts['model_dir']
                self._processor = CLIPProcessor.from_pretrained(model_dir)
                self._model = CLIPModel.from_pretrained(model_dir)
                # self._processor.to(_map_location)
                # self._model.to(_map_location)

                print("Model loaded successfully")
            except Exception as e:
                print("Failed to load the the model.")
                print(e)
                raise
        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """Perform inference on the input data.

        :param input_data: Input images for prediction.
        :type input_data: Pandas DataFrame with a first column name ["image"] of images where each
        image is in base64 String format.
        :param task: Task type of the model.
        :type task: HFTaskLiterals
        :param tokenizer: Preprocessing configuration loader.
        :type tokenizer: transformers.AutoImageProcessor
        :param model: Pytorch model weights.
        :type model: transformers.AutoModelForImageClassification
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns ["filename", "probs", "labels"] for classification and
        ["filename", "boxes"] for object detection, instance segmentation
        """
        # Decode the base64 image column
        decoded_images = input_data.loc[
            :, [MLFlowSchemaLiterals.INPUT_COLUMN_IMAGE]
        ].apply(axis=1, func=process_image)

        captions = input_data['text'].iloc[0].split(',')

        # To Do: change image height and width based on kwargs.

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
                MLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS,
                MLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS,
            ]
        )

        labels = [captions] * len(conf_scores)
        df_result[MLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS], df_result[MLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS]\
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
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        return probs