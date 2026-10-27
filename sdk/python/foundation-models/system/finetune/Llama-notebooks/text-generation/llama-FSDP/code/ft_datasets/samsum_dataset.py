# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
import numpy as np


def get_preprocessed_samsum(dataset_config, tokenizer, split):
    if split == dataset_config.prediction_split:
        dataset = datasets.load_dataset("samsum", split=dataset_config.validation_split)
    else:
        dataset = datasets.load_dataset("samsum", split=split)

    train_prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )
    val_prompt = f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"

    def train_apply_prompt_template(sample):
        return {
            "text": train_prompt.format(
                dialog=sample["dialogue"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    def val_apply_prompt_template(sample):
        return {"text": val_prompt.format(dialog=sample["dialogue"])}

    if split == dataset_config.train_split:
        dataset = dataset.map(
            train_apply_prompt_template, remove_columns=list(dataset.features)
        )
    else:
        dataset = dataset.map(
            val_apply_prompt_template, remove_columns=list(dataset.features)
        )

    def add_label_ids(sample):
        sample["labels"] = sample["input_ids"]
        return sample

    if split == dataset_config.prediction_split:
        dataset = dataset.map(
            lambda sample: tokenizer(sample["text"]),
            batched=True,
            remove_columns=list(dataset.features),
        )
        dataset = dataset.select(np.arange(300))
    else:
        dataset = dataset.map(
            lambda sample: tokenizer(
                sample["text"],
                truncation=True,
                padding="max_length",
                max_length=dataset_config.max_input_length,
            ),
            batched=True,
            remove_columns=list(dataset.features),
        )
        dataset = dataset.map(lambda sample: add_label_ids(sample), batched=True)

    return dataset
