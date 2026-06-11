# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    prediction_split: str = "test"
    input_length: int = 2048
    dataset_path: str = "dummy_path"  # Needs to be supplied via command line argument
    train_filename: str = "samsum-train.jsonl"
    test_filename: str = "samsum-validation.jsonl"
    max_input_length: str = 512  # Maximum number of input tokens
    max_gen_length: str = 100  # Number of tokens to be generated


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass
class emotion_detection_dataset:
    dataset: str = "emotion_detection_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    max_input_length: int = 512
    num_labels: int = 6


class bing_text_classification_dataset:
    dataset: str = "bing_text_classification_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    dataset_path: str = "dummy_path"  # Needs to be supplied via command line argument
    train_filename: str = "FY22H2_data_llama_fomrat_lpsat_without_body.jsonl"
    test_filename: str = "test.jsonl"
    columns_to_be_removed = ["instruction"]
    max_input_length: int = 4096
    num_labels: int = 21
