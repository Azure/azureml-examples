# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""
GRPO trainer
"""
import logging
import os
import sys
from dataclasses import dataclass, field

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)

import datasets
from datasets import DatasetDict, load_dataset
from grpo_trainer_callbacks import SaveMLflowModelCallback
from grpo_trainer_rewards import get_rewards_funcs

# VLLM_PP_LAYER_PARTITION = layers per pipeline stage
# VLLM_PP_NUM_PARTITIONS = number of pipeline stages (GPUs/processes)
# Both are essential for configuring pipeline parallelism in vLLM for efficient distributed training or inference.
os.environ["VLLM_PP_LAYER_PARTITION"] = "28"
os.environ["VLLM_PP_NUM_PARTITIONS"] = "8"

logger = logging.getLogger(__name__)


# System prompt used at the time of sampling
system_prompt = "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags."

# Chat template used for the tokenizer
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Extra script arguments for the GRPO training script.
    """

    final_model_save_path: str = field(
        default="final_model", metadata={"help": "Path to save the final model."}
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "accuracy"],
        metadata={
            "help": "List of reward functions. Possible values: 'format', 'accuracy'."
        },
    )
    mlflow_task_type: str = field(default="chat-completion")
    base_model_name: str = field(
        default="base_model", metadata={"help": "Base model name for MLflow."}
    )


def get_tokenizer(model_args: ModelConfig) -> PreTrainedTokenizer:
    """Returns the tokenizer for the model.

    Args:
        model_args (ModelConfig): Model configuration object.
    Returns:
        PreTrainedTokenizer: The tokenizer for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return tokenizer


def make_conversation(example, system_prompt=None):
    """Transform the given record to be compatible for GRPO training.

    Args:
        example (dict): The input record.
        system_prompt (str): The system prompt to be used.
    Returns:
        dict: The transformed record.
    """
    prompt = []

    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})

    prompt.append({"role": "user", "content": example["problem"]})
    return {"prompt": prompt}


def prepare_dataset(dataset_folder, system_prompt=None):
    """Load the splits from the given dataset folder and transform the dataset to be compatible for GRPO training.

    Args:
        dataset_folder (str): The folder containing the dataset splits.
                              Scans for train.jsonl, validation.jsonl, and test.jsonl.
        system_prompt (str): The system prompt to be used.
    Returns:
        DatasetDict: The transformed dataset.
    """
    split_names = ["train", "validation", "test"]
    dataset_dict = {}
    for split in split_names:
        temp_path = os.path.join(dataset_folder, f"{split}.jsonl")
        if os.path.exists(temp_path):
            dataset_dict[split] = load_dataset(
                "json",
                data_files=temp_path,
                split="train",
            )

    dataset = DatasetDict(dataset_dict)
    dataset = dataset.map(
        lambda example: make_conversation(example, system_prompt=system_prompt)
    )
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    return dataset


def main(script_args, training_args, model_args):
    """Main function to run the GRPO training script.

    Args:
        script_args (GRPOScriptArguments): Arguments to configure the datasets, reward functions.
        training_args (GRPOConfig): Trainer specific settings such as vllm server config,
                                    learning rate and reward weights.
        model_args (ModelConfig): Arguments to load the model.
    Returns:
        None
    """
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)

    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    dataset_path = script_args.dataset_name
    current_policy = model_args.model_name_or_path

    # Load tokenizer
    tokenizer = get_tokenizer(model_args)

    # Load the dataset
    dataset = prepare_dataset(dataset_path, system_prompt=system_prompt)
    logger.info(dataset)

    # Load the reward functions
    reward_function = get_rewards_funcs(script_args.reward_funcs)

    # Add save callback
    save_mlflow_callback = SaveMLflowModelCallback(
        mlflow_model_save_path=script_args.final_model_save_path,
        mlflow_task_type=script_args.mlflow_task_type,
        base_model_name=script_args.base_model_name,
        processing_class=tokenizer,
    )

    # Create the GRPOTrainer (It does SAMPLING, GRADING and TRAINING)
    trainer = GRPOTrainer(
        # The model to be trained, same copy of the model is used as reference policy.
        model=current_policy,
        # Rewards functions to be used by graders defined in "grpo_trainer_rewards.py".
        reward_funcs=reward_function,
        args=training_args,
        # Each prompt from the dataset is used to generate multiple samples.
        train_dataset=dataset[script_args.dataset_train_split],
        # Configuration for lora.
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        callbacks=[save_mlflow_callback],
    )
    # Trigger the training loop
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
