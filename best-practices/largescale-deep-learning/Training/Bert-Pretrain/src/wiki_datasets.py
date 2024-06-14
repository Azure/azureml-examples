# -------------------------------------------------------------------------
# Portions Copyright (c) Microsoft Corporation.  All rights reserved.

# --------------------------------------------------------------------------
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union, Dict, Callable
from datasets import load_dataset, load_metric, interleave_datasets
from datasets import DatasetDict, Dataset, Metric  # used for typing
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from itertools import chain 

def construct_tokenizer_function(
    tokenizer: PreTrainedTokenizerBase
) -> Callable[[Union[Dict, Any]], Union[Dict, Any]]:
    """Construct function used to tokenize WIKI data.
    """
    max_length = 512
    def tokenize_function(examples: Union[Dict, Any]) -> Dict:
        return tokenizer(
            examples["text"], return_special_tokens_mask=True, truncation=True, max_length=max_length
        )
    return tokenize_function

def load_raw_wiki_dataset() -> Union[DatasetDict, Dataset]:
    wiki_data = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    wiki_data = wiki_data.remove_columns([col for col in wiki_data.column_names if col != "text"])
    return wiki_data

def load_raw_corpus_dataset() -> Union[DatasetDict, Dataset]:
    corpus_data = load_dataset("bookcorpus", split="train", streaming=True)
    return corpus_data

def load_encoded_wiki_dataset(
    tokenizer: PreTrainedTokenizerBase
) -> Union[DatasetDict, Dataset]:

    """Load wiki + corpus data, apply tokenizer and split into train/validation."""
    # Construct tokenizer function
    tokenizer_func = construct_tokenizer_function(tokenizer=tokenizer)
    # load raw data
    raw_wiki_dataset = load_raw_wiki_dataset()
    raw_corpus_dataset = load_raw_corpus_dataset()
    assert raw_corpus_dataset.features.type == raw_wiki_dataset.features.type
    # Combine datasets
    full_raw_dataset = interleave_datasets([raw_corpus_dataset, raw_wiki_dataset])
    # tokenize dataset
    encoded_dataset = full_raw_dataset.map(tokenizer_func, batched=True, remove_columns=["text"])
    # Prepare function for grouping text into batches
    group_texts=construct_group_texts(tokenizer=tokenizer)
    # Batch the data
    encoded_dataset = encoded_dataset.map(group_texts, batched=True)
    return encoded_dataset

def load_metric_from_wiki() -> Metric:
    """Load the metric."""
    metric = load_metric("accuracy")
    return metric


# Main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length. #From: https://github.com/philschmid/deep-learning-habana-huggingface/blob/master/pre-training/pre-training-bert.ipynb
def construct_group_texts(
    tokenizer: PreTrainedTokenizerBase
) -> Callable[[Union[Dict, Any]], Union[Dict, Any]]:
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= tokenizer.model_max_length:
            total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    return group_texts
