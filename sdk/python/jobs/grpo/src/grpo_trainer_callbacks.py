# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""
callback utilities
"""
import logging
import shutil
from pathlib import Path
from typing import Union

import azureml.evaluate.mlflow as mlflow
from mlflow.models import Model
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def copy_tokenizer_files_to_model_folder(mlflow_model_folder: str):
    """Copy tokenizer files to model folder.
    The mlflow model format expects tokenizer and model files in model folder i.e. "data/model".

    Args:
        mlflow_model_folder (str): Path to the mlflow model folder.

    Returns:
        None
    """
    src_dir = Path(mlflow_model_folder, "data", "tokenizer")
    dst_dir = Path(mlflow_model_folder, "data", "model")
    if src_dir.is_dir() and dst_dir.is_dir():
        logger.info("Copying tokenizer files to model folder")
        shutil.copytree(
            str(Path(mlflow_model_folder, "data", "tokenizer")),
            str(Path(mlflow_model_folder, "data", "model")),
            dirs_exist_ok=True,
        )
        logger.info("Copy completed for tokenizer files to model folder")
    else:
        logger.warning("Couldn't copy the tokenizer files to model folder.")


class SaveMLflowModelCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that saves the model in mlflow model format.
    """

    def __init__(
        self,
        mlflow_model_save_path: Union[str, Path],
        mlflow_task_type: str,
        base_model_name: str,
        **kwargs,
    ) -> None:

        self.mlflow_model_save_path = mlflow_model_save_path
        self.mlflow_task_type = mlflow_task_type
        self.metadata = {
            "azureml.base_image": "mcr.microsoft.com/azureml/curated/foundation-model-inference:71",
            "base_model_asset_id": "",
            "base_model_name": base_model_name,
            "base_model_task": mlflow_task_type,
            "finetuning_task": mlflow_task_type,
            "is_acft_model": True,
            "is_finetuned_model": True,
        }

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Callback function to save the model in mlflow model format at the end of the training.

        Args:
            args (TrainingArguments): The training arguments.
            state (TrainerState): The trainer state.
            control (TrainerControl): The trainer control.
            **kwargs: Additional keyword arguments. The kwargs contain the model and tokenizer to be saved.
                - model: The model to be saved.
                - processing_class: The tokenizer to be used for saving the model.
        Returns:
            None
        """
        full_path = self.mlflow_model_save_path
        model, tokenizer = kwargs["model"], kwargs["processing_class"]

        # Saving the mlflow on world process 0
        if state.is_world_process_zero:
            hf_conf = {"task_type": self.mlflow_task_type}
            # This is to unify hfv2/OSS metadata dump
            mlflow_model = Model(metadata=self.metadata)
            mlflow.hftransformers.save_model(
                model,
                full_path,
                tokenizer,
                model.config,
                mlflow_model=mlflow_model,
                hf_conf=hf_conf,
            )
            # Copy tokenizer files to model folder
            copy_tokenizer_files_to_model_folder(full_path)
            logger.info(f"MLflow model saved at {full_path}")
