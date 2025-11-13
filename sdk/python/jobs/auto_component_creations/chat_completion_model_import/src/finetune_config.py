# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing utilities for loading finetune config."""

import copy
import json
from pathlib import Path
from typing import Optional, Dict, Any

from azureml.acft.accelerator.constants import LoraAlgo
from azureml.acft.contrib.hf.nlp.utils.common_utils import deep_update
from azureml.acft.contrib.hf.nlp.constants.constants import HfModelTypes

from azureml.acft.common_components import get_logger_app


logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.model_selector.finetune_config")


# TODO - Move REFINED_WEB to :dataclass HfModelTypes
REFINED_WEB = "RefinedWeb"
MIXFORMER_SEQUENTIAL = "mixformer-sequential"  # Phi models


# user config passed along with model, will be prefered over default settings
# NOTE Deleted `load_model_kwargs` for trust_remote_code=True as for falcon models
# we are adding a hack and overriding :meth `from_pretrained` for Text, Token and
# QnA auto classes in finetune.py.
# Don't forget to add back the `load_model_kwargs` once the hack to override methods
# is removed
ACFT_CONFIG = {
    "tiiuae/falcon-7b": {
        "load_config_kwargs": {
            "trust_remote_code": True,
        },
        "load_tokenizer_kwargs": {
            # adding eos_token as pad_token. The value of eos_token is taken from tokenization_config.json file
            "pad_token": "<|endoftext|>",
            "trust_remote_code": True,
        },
        "lora_algo": LoraAlgo.PEFT,
        "mlflow_ft_conf": {
            "mlflow_hftransformers_misc_conf": {
                "config_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                "tokenizer_hf_load_kwargs": {
                    "model_input_names": ["input_ids", "attention_mask"],
                    "return_token_type_ids": False,
                },
                "model_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                # "tokenizer_config": {
                #     "return_token_type_ids": False,
                # },
            },
            "mlflow_save_model_kwargs": {
                "extra_pip_requirements": ["einops"],
            },
        },
    },
    "tiiuae/falcon-40b": {
        "load_config_kwargs": {
            "trust_remote_code": True,
        },
        "load_tokenizer_kwargs": {
            # adding eos_token as pad_token. The value of eos_token is taken from tokenization_config.json file
            "pad_token": "<|endoftext|>",
            "trust_remote_code": True,
        },
        "lora_algo": LoraAlgo.PEFT,
        "mlflow_ft_conf": {
            "mlflow_hftransformers_misc_conf": {
                "config_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                "tokenizer_hf_load_kwargs": {
                    "model_input_names": ["input_ids", "attention_mask"],
                    "return_token_type_ids": False,
                },
                "model_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                # "tokenizer_config": {
                #     "return_token_type_ids": False,
                # },
            },
            "mlflow_save_model_kwargs": {
                "extra_pip_requirements": ["einops"],
            },
        },
    },
    HfModelTypes.REFINEDWEBMODEL: {
        "load_config_kwargs": {
            "trust_remote_code": True,
        },
        "load_tokenizer_kwargs": {
            # adding eos_token as pad_token. The value of eos_token is taken from tokenization_config.json file
            "pad_token": "<|endoftext|>",
            "trust_remote_code": True,
        },
        "lora_algo": LoraAlgo.PEFT,
        "mlflow_ft_conf": {
            "mlflow_hftransformers_misc_conf": {
                "config_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                "tokenizer_hf_load_kwargs": {
                    "model_input_names": ["input_ids", "attention_mask"],
                    "return_token_type_ids": False,
                },
                "model_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                # "tokenizer_config": {
                #     "return_token_type_ids": False,
                # },
            },
            "mlflow_save_model_kwargs": {
                "extra_pip_requirements": ["einops"],
            },
        },
    },
    HfModelTypes.FALCON: {
        "load_config_kwargs": {
            "trust_remote_code": True,
        },
        "load_tokenizer_kwargs": {
            # adding eos_token as pad_token. The value of eos_token is taken from tokenization_config.json file
            "pad_token": "<|endoftext|>",
            "trust_remote_code": True,
        },
        "lora_algo": LoraAlgo.PEFT,
        "mlflow_ft_conf": {
            "mlflow_hftransformers_misc_conf": {
                "config_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                "tokenizer_hf_load_kwargs": {
                    "model_input_names": ["input_ids", "attention_mask"],
                    "return_token_type_ids": False,
                },
                "model_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                # "tokenizer_config": {
                #     "return_token_type_ids": False,
                # },
            },
            "mlflow_save_model_kwargs": {
                "extra_pip_requirements": ["einops"],
            },
        },
    },
    REFINED_WEB: {
        "load_config_kwargs": {
            "trust_remote_code": True,
        },
        "load_tokenizer_kwargs": {
            # adding eos_token as pad_token. The value of eos_token is taken from tokenization_config.json file
            "pad_token": "<|endoftext|>",
            "trust_remote_code": True,
        },
        "lora_algo": LoraAlgo.PEFT,
        "mlflow_ft_conf": {
            "mlflow_hftransformers_misc_conf": {
                "config_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                "tokenizer_hf_load_kwargs": {
                    "model_input_names": ["input_ids", "attention_mask"],
                    "return_token_type_ids": False,
                },
                "model_hf_load_kwargs": {
                    "trust_remote_code": True,
                },
                # "tokenizer_config": {
                #     "return_token_type_ids": False,
                # },
            },
            "mlflow_save_model_kwargs": {
                "extra_pip_requirements": ["einops"],
            },
        },
    },
    HfModelTypes.LLAMA: {
        "load_tokenizer_kwargs": {
            "add_eos_token": True,
            "padding_side": "right",
        },
        "lora_algo": LoraAlgo.PEFT,
        "mlflow_ft_conf": {
            "mlflow_hftransformers_misc_conf": {
                "tokenizer_hf_load_kwargs": {
                    "add_eos_token": True,
                    "padding_side": "right",
                },
            }
        },
    },
}


# The constant dictionary is the legacy way of storing finetune config. The new versions of models
# have the finetune config packaged with the mlflow model artifacts.
FINETUNE_CONFIG_V0 = ACFT_CONFIG


class FinetuneConfig:
    """Class for finetune config utilities."""

    # If version is not mentioned in the json, it falls under v1 config
    _DEFAULT_VERSION = "1"

    def __init__(
        self,
        task_name: str,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        artifacts_finetune_config_path: Optional[str] = None,
        io_finetune_config_path: Optional[str] = None,
    ) -> None:
        """
        Finetune config init.

        :param task_name - The finetune task
        :type str
        :param model_name - The huggingface model name
        :type Optional[str]
        :param model_type - The name of the model family. For instance, the model names llama-2-7b, llama-2-13b,
        llama-2-70b belong to model_type llama
        :type Optional[str]
        :param artifacts_finetune_config_path - The path to finetune config packaged with model artifacts. The
        finetune config comes with 2 versions. More details about this is defined in the respective methods
        :type Optional[str]
        :param io_finetune_config_path - The io path to the finetune config. This takes precedence over the config
        passed with model artifacts.
        :type Optional[str]
        """
        self.task_name = task_name
        self.model_name = model_name
        self.model_type = model_type
        self.artifacts_finetune_config_path = artifacts_finetune_config_path
        self.io_finetune_config_path = io_finetune_config_path

    def _read_ft_config_json_file(
        self, json_file_path: Optional[str]
    ) -> Dict[str, Any]:
        """Read finetune config json file.

        :param json_file_path - Path to the finetune config json file
        :type Optional[str]
        :return finetune_config
        :rtype Dict[str, Any]
        """
        if not json_file_path or not Path(json_file_path).is_file():
            return {}

        with open(json_file_path, "r", encoding="utf-8") as rptr:
            json_data = json.load(rptr)
        # filter the finetune config for version 2
        if json_data.get("version", self._DEFAULT_VERSION) == "2":
            json_data = json_data.get("task_specific_config", {}).get(
                self.task_name, {}
            )

        return json_data

    def get_finetune_config(self) -> Dict[str, Any]:
        """Fetch the finetune config. Attempts to read the finetune config with v2 version."""
        finetune_config_artifacts = self._read_ft_config_json_file(
            self.artifacts_finetune_config_path
        )
        finetune_config_io = self._read_ft_config_json_file(
            self.io_finetune_config_path
        )
        # combine artifacts and io finetune config.
        finetune_config = deep_update(
            finetune_config_artifacts,
            finetune_config_io,  # finetune_config_io is given more precedence over finetune_config_artifacts
        )

        # read legacy config from FINETUNE_CONFIG_V0
        if not finetune_config:
            finetune_config = self._read_legacy_finetune_config()
        logger.info(
            f"Setting the following finetune config to this model: {finetune_config}"
        )
        return finetune_config

    def _read_legacy_finetune_config(self) -> Dict[str, Any]:
        """Read the finetune config from the legacy ACFT_CONFIG a.k.a FINETUNE_CONFIG_V0.

        If both model_name and model_type are present, precedence is given to model_name over model_type.
        """
        finetune_config_v0 = {}
        if self.model_name is not None and self.model_name in FINETUNE_CONFIG_V0:
            finetune_config_v0 = copy.deepcopy(FINETUNE_CONFIG_V0[self.model_name])
            logger.info(
                f"Found model name in finetune config version v0 - {finetune_config_v0}"
            )
        elif self.model_type is not None and self.model_type in ACFT_CONFIG:
            finetune_config_v0 = copy.deepcopy(FINETUNE_CONFIG_V0[self.model_type])
            logger.info(
                f"Found model type in finetune config version v0 - {finetune_config_v0}"
            )
        return finetune_config_v0
