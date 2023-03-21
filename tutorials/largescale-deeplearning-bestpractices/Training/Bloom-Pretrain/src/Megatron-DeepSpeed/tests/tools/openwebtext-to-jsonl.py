#!/usr/bin/env python

# generate a jsonl version of a small slice of a dataset that can be fed to megatron-lm preprocessor

import sys
from datasets import load_dataset

dataset_name = "stas/openwebtext-10k"

# subset to jsonlines
n_samples = 1000
ds = load_dataset(dataset_name, split='train')
ds_small = ds.select(range(n_samples))
path = f"openwebtext-{n_samples}.jsonl"
ds_small.to_json(path, orient="records", lines=True)
