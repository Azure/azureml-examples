# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for model selector component."""
import os
import shutil
from pathlib import Path
import argparse
import json
from argparse import Namespace
import copy
import yaml
import logging
from typing import Optional, Dict, Any

from transformers.utils import GENERATION_CONFIG_NAME
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from azureml.acft.accelerator.utils.code_utils import update_json_file_and_overwrite

from azureml.acft.contrib.hf.nlp.task_factory import get_task_runner
from azureml.acft.contrib.hf.nlp.utils.common_utils import deep_update
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.contrib.hf.nlp.constants.constants import SaveFileConstants, MLFlowHFFlavourConstants

from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME

from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore

from finetune_config import FinetuneConfig


logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.model_selector.model_selector")

COMPONENT_NAME = "ACFT-Model_import"

# TODO - Move REFINED_WEB to :dataclass HfModelTypes
REFINED_WEB = "RefinedWeb"
MIXFORMER_SEQUENTIAL = "mixformer-sequential"  # Phi models


# TODO Move this constants class to package
class ModelSelectorConstants:
    """Model import constants."""

    ASSET_ID_NOT_FOUND = "ASSET_ID_NOT_FOUND"
    MODEL_NAME_NOT_FOUND = "MODEL_NAME_NOT_FOUND"


FLAVOR_MAP = {
    # OSS Flavor
    "transformers": {
        "tokenizer": "components/tokenizer",
        "model": "model",
        "config": "model"
    },
    "hftransformersv2": {
        "tokenizer": "data/tokenizer",
        "model": "data/model",
        "config": "data/config"
    },
    "hftransformers": {
        "tokenizer": "data/tokenizer",
        "model": "data/model",
        "config": "data/config"
    }
}


def get_model_asset_id() -> str:
    """Read the model asset id from the run context.

    TODO Move this function to run utils
    """
    try:
        from azureml.core import Run

        run_ctx = Run.get_context()
        if isinstance(run_ctx, Run):
            run_details = run_ctx.get_details()
            return run_details['runDefinition']['inputAssets']['mlflow_model_path']['asset']['assetId']
        else:
            logger.info("Found offline run")
            return ModelSelectorConstants.ASSET_ID_NOT_FOUND
    except Exception as e:
        logger.info(f"Could not fetch the model asset id: {e}")
        return ModelSelectorConstants.ASSET_ID_NOT_FOUND


def validate_huggingface_id(huggingface_id: str) -> None:
    """Validate the huggingface_id using Hfapi. Raise exception if the huggingface id is invalid."""
    from huggingface_hub import HfApi
    hf_api = HfApi()  # by default endpoint points to https://huggingface.co

    try:
        model_infos = [
            info
            for info in hf_api.list_models(model_name=huggingface_id)
            if info.modelId == huggingface_id
        ]
    except ConnectionError:
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "Passing huggingface ID is only valid with valid internet connection."
                )
            )
        )

    if not model_infos:
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    f"Invalid hugggingface ID found: {huggingface_id}. Please fix it and try again."
                )
            )
        )
    return True


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description="Model selector for hugging face models", allow_abbrev=False)

    parser.add_argument(
        "--output_dir",
        default="model_selector_output",
        type=str,
        help="folder to store model selector outputs",
    )

    parser.add_argument(
        "--huggingface_id",
        default=None,
        type=str,
        help="Input HuggingFace model id takes priority over model_id.",
    )

    # Task settings
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model id used to load model checkpoint.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="SingleLabelClassification",
        help="Task Name",
    )

    # Continual Finetuning
    parser.add_argument(
        "--pytorch_model_path",
        default=None,
        type=str,
        help="model path containing pytorch model"
    )
    parser.add_argument(
        "--mlflow_model_path",
        default=None,
        type=str,
        help="model path containing mlflow model"
    )

    # FT config
    parser.add_argument(
        "--finetune_config_path",
        default=None,
        type=str,
        help="finetune config file path"
    )

    return parser


def model_selector(args: Namespace) -> Dict[str, Any]:
    """Model selector main.

    :param args - component args
    :type Namespace
    :return Meta data saved by model selector component (model selector args)
    :rtype Dict[str, Any]
    """
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.huggingface_id is not None:
        # remove the spaces at either ends of hf id
        args.model_name = args.huggingface_id.strip()
        # validate hf_id
        if validate_huggingface_id(args.model_name):
            # if neither pytorch nor mlflow model path is provided, pull from HF
            if args.pytorch_model_path is None and args.mlflow_model_path is None:

                model_dir = Path(args.output_dir)
                # download config, tokenizer, and model into model_dir
                # Download config, tokenizer, and model from Hugging Face
                config = AutoConfig.from_pretrained(args.model_name)
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
                model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)

                # Save everything into args.output_dir
                config.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                model.save_pretrained(model_dir)
                logger.info(f"Downloaded and saved model {args.model_name} to {model_dir}")

                # Clear the Hugging Face cache
                hf_cache = os.getenv(
                    "HF_HOME",
                    os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
                )
                shutil.rmtree(hf_cache, ignore_errors=True)
                logger.info(f"Cleared Hugging Face cache at {hf_cache}")
    else:
        # TODO Revist whether `model_id` is still relevant
        args.model_name = args.model_id

    # Add the model asset id to model_selector_args
    model_asset_id = get_model_asset_id()
    logger.info(f"Model asset id: {model_asset_id}")
    setattr(args, "model_asset_id", model_asset_id)

    task_runner = get_task_runner(task_name=args.task_name)()
    task_runner.run_modelselector(**vars(args))

    # read model selector args
    model_selector_args_save_path = Path(args.output_dir, SaveFileConstants.MODEL_SELECTOR_ARGS_SAVE_PATH)
    with open(model_selector_args_save_path, "r") as rptr:
        model_selector_args = json.load(rptr)

    return model_selector_args


def fetch_model_type(model_path: str) -> Optional[str]:
    """Fetch the model type.

    :param model_path - path to the model artifacts
    :type str
    :return model type
    :rtype Optional[str]
    """
    model_type = None
    try:
        # fetch model_type
        model_config_path = Path(model_path, "config.json")
        if model_config_path.is_file():
            with open(model_config_path, "r") as fp:
                model_config = json.load(fp)
                model_type = model_config.get("model_type", None)
        else:
            logger.info(f"Model config.json does not exist for {model_path}")
    except Exception:
        logger.info(f"Unable to fetch model_type for {model_path}")

    return model_type


def read_base_model_finetune_config(mlflow_model_path: str, task_name: str) -> Dict[str, Any]:
    """Read the finetune config from base model.

    :param mlflow_model_path - Path to the mlflow model
    :type str
    :param task_name - Finetune task
    :type str
    :return base model finetune config
    :type Optional[Dict[str, Any]]
    """
    if mlflow_model_path is None:
        return {}

    mlflow_config_file = Path(mlflow_model_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE)
    mlflow_ftconf_data = {}
    if mlflow_config_file.is_file():
        # pass mlflow data to ft config if available
        mlflow_data = None
        try:
            with open(str(mlflow_config_file), "r") as fp:
                mlflow_data = yaml.safe_load(fp)
            if mlflow_data and "flavors" in mlflow_data:
                for key in mlflow_data["flavors"]:
                    if key in FLAVOR_MAP:
                        for key2 in mlflow_data["flavors"][key]:
                            if key2 == "generator_config" and task_name == "TextGeneration":
                                generator_config = mlflow_data["flavors"][key]["generator_config"]
                                mlflow_ftconf_data_temp = {
                                        "load_config_kwargs": copy.deepcopy(generator_config),
                                        "update_generator_config": copy.deepcopy(generator_config),
                                        "mlflow_ft_conf": {
                                            "mlflow_hftransformers_misc_conf": {
                                                "generator_config": copy.deepcopy(generator_config),
                                            },
                                        },
                                    }
                                mlflow_ftconf_data = deep_update(mlflow_ftconf_data_temp, mlflow_ftconf_data)
                            elif key2 == "model_hf_load_kwargs":
                                model_hf_load_kwargs = mlflow_data["flavors"][key]["model_hf_load_kwargs"]
                                mlflow_ftconf_data_temp = {
                                        "mlflow_ft_conf": {
                                            "mlflow_hftransformers_misc_conf": {
                                                "model_hf_load_kwargs": copy.deepcopy(model_hf_load_kwargs),
                                            },
                                        },
                                    }
                                mlflow_ftconf_data = deep_update(mlflow_ftconf_data_temp, mlflow_ftconf_data)
        except Exception:
            logger.info("Error while updating base model finetune config from MlModel file.")
    else:
        logger.info("MLmodel file does not exist")

    return mlflow_ftconf_data


@swallow_all_exceptions(time_delay=60)
def main():
    """Parse args and import model."""
    # args
    parser = get_parser()
    args, _ = parser.parse_known_args()

    set_logging_parameters(
        task_type=args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )

    # Validated custom model type
    if args.mlflow_model_path and \
       not Path(args.mlflow_model_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE).is_file():
        raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            "MLmodel file is not found, If this is a custom model "
                            "it needs to be connected to pytorch_model_path"
                        )
                    )
            )

    # Adding flavor map to args
    setattr(args, "flavor_map", FLAVOR_MAP)

    if args.pytorch_model_path:
        logger.info(f"Using PyTorch model path: {args.pytorch_model_path}")
        model_artifact_dir = Path(args.pytorch_model_path) / "model_artifact" / "model"
        if model_artifact_dir.is_dir():
            args.pytorch_model_path = str(model_artifact_dir)
            # To support the latest pytorch artifact path
            logger.info(f"Updated PyTorch model path: {args.pytorch_model_path}")
        else:
            logger.info(f"'model' subfolder does not exist, using original path: {args.pytorch_model_path}")

    # run model selector
    model_selector_args = model_selector(args)
    model_name = model_selector_args.get("model_name", ModelSelectorConstants.MODEL_NAME_NOT_FOUND)
    logger.info(f"Model name - {model_name}")
    logger.info(f"Task name: {getattr(args, 'task_name', None)}")
    # Validate port for right model type
    if args.pytorch_model_path and Path(args.pytorch_model_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE).is_file():
        raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        "MLFLOW model is connected to pytorch_model_path, "
                        "it needs to be connected to mlflow_model_path"
                    )
                )
            )

    # load ft config and update ACFT config
    # finetune_config_dict = load_finetune_config(args)
    ft_config_obj = FinetuneConfig(
        task_name=args.task_name,
        model_name=model_name,
        model_type=fetch_model_type(str(Path(args.output_dir, model_name))),
        artifacts_finetune_config_path=str(
            Path(
                args.pytorch_model_path or args.mlflow_model_path or "",
                SaveFileConstants.ACFT_CONFIG_SAVE_PATH
            )
        ),
        io_finetune_config_path=args.finetune_config_path
    )
    finetune_config = ft_config_obj.get_finetune_config()

    # read finetune config from base mlmodel file
    # Priority order: io_finetune_config > artifacts_finetune_config > base_model_finetune_config
    updated_finetune_config = deep_update(
        read_base_model_finetune_config(
            args.mlflow_model_path,
            args.task_name
        ),
        finetune_config
    )

    # Copy Mlmodel generator config so that FTed model also uses same generator config while evaluation.
    # (Settings like `max_new_tokens` can help us reduce inference time.)
    # We are updating generation_config.json so that no conflicts will be present between
    # model's config and model's generator_config. (If there is conflict we get warning in logs
    # and from transformers>=4.41.0 exceptions will be raised if `_from_model_config` key is present.)
    if "update_generator_config" in updated_finetune_config:
        generator_config = updated_finetune_config.pop("update_generator_config")
        base_model_generation_config_file = Path(
            args.output_dir, model_selector_args["model_name"], GENERATION_CONFIG_NAME
        )
        if base_model_generation_config_file.is_file():
            update_json_file_and_overwrite(str(base_model_generation_config_file), generator_config)
            logger.info(f"Updated {GENERATION_CONFIG_NAME} with {generator_config}")
        else:
            logger.info(f"Could not update {GENERATION_CONFIG_NAME} as not present.")
    else:
        logger.info(f"{MLFlowHFFlavourConstants.MISC_CONFIG_FILE} does not have any generation config parameters.")

    logger.info(f"Updated finetune config with base model config: {updated_finetune_config}")
    # save FT config
    with open(str(Path(args.output_dir, SaveFileConstants.ACFT_CONFIG_SAVE_PATH)), "w") as rptr:
        json.dump(updated_finetune_config, rptr, indent=2)
    logger.info(f"Saved {SaveFileConstants.ACFT_CONFIG_SAVE_PATH}")

    # copy the mlmodel file to output dir. This is only applicable for mlflow model
    if args.mlflow_model_path is not None:
        mlflow_config_file = Path(args.mlflow_model_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE)
        if mlflow_config_file.is_file():
            shutil.copy(str(mlflow_config_file), args.output_dir)
            logger.info(f"Copied {MLFlowHFFlavourConstants.MISC_CONFIG_FILE} file to output dir.")

        # copy conda file
        conda_file_path = Path(args.mlflow_model_path, MLFlowHFFlavourConstants.CONDA_YAML_FILE)
        if conda_file_path.is_file():
            shutil.copy(str(conda_file_path), args.output_dir)
            logger.info(f"Copied {MLFlowHFFlavourConstants.CONDA_YAML_FILE} file to output dir.")

        # copy inference config files
        mlflow_ml_configs_dir = Path(args.mlflow_model_path, "ml_configs")
        ml_config_dir = Path(args.output_dir, "ml_configs")
        if mlflow_ml_configs_dir.is_dir():
            shutil.copytree(
                mlflow_ml_configs_dir,
                ml_config_dir
            )
            logger.info("Copied ml_configs folder to output dir.")


if __name__ == "__main__":
    main()
