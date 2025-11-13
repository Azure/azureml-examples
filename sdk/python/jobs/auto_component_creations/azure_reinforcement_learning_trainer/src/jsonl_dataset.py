# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Custom JSONL Dataset for VERL GRPO Training."""

import copy
import logging
import os
from typing import Optional

import datasets
from omegaconf import DictConfig, ListConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class JSONLDataset(RLHFDataset):
    """
    Load and preprocess RLHF data from JSONL files.
    
    Inherits from RLHFDataset but overrides the file reading logic
    to support JSONL format instead of Parquet format.
    
    Args:
        data_files (str or list): Path(s) to JSONL file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def _detect_format(self, file_path: str) -> str:
        """Detect data format from file extension.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Format string for datasets.load_dataset
        """
        file_lower = file_path.lower()
        if file_lower.endswith('.parquet'):
            return 'parquet'
        elif file_lower.endswith(('.json', '.jsonl')):
            return 'json'
        elif file_lower.endswith('.csv'):
            return 'csv'
        else:
            logger.warning(f"Unknown extension for '{file_path}', assuming parquet")
            return 'parquet'

    def _read_files_and_tokenize(self):
        """Override to support multiple file formats including JSONL."""
        dataframes = []
        for data_file in self.data_files:
            # Detect format from file extension
            format_type = self._detect_format(data_file)
            logger.info(f"Loading '{data_file}' as {format_type} format")
            
            try:
                dataframe = datasets.load_dataset(format_type, data_files=data_file)["train"]
                dataframes.append(dataframe)
                logger.info(f"Successfully loaded {len(dataframe)} samples from {data_file}")
            except Exception as e:
                logger.error(f"Error loading {data_file}: {e}")
                raise

        self.dataframe = datasets.concatenate_datasets(dataframes)
        logger.info(f"Total dataset size: {len(self.dataframe)} samples")
        
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        logger.info(f"After filtering: {len(self.dataframe)} samples")
