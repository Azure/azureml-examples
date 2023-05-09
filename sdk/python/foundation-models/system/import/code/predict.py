# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import torch
import numpy as np
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM


logger = logging.getLogger(__name__)

model = None
tokenizer = None

def init():
    global model
    global tokenizer

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), "INPUT_model_path")
    # subfolder = "open_llama_7b_preview_200bt_transformers_weights"
    try:
        logger.info("Loading model from path.")
        # tokenizer = LlamaTokenizer.from_pretrained(model_path, subfolder=subfolder, use_fast=False, local_files_only=True)
        # model = LlamaForCausalLM.from_pretrained(model_path, subfolder=subfolder, local_files_only=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
        model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
        logger.info("Loading successful.")
    except Exception as e:
        return json.dumps({"error": str(e)})


def run(data, **kwargs):
    if not model or not tokenizer:
        return json.dumps({"error": "Model or tokenizer was not initialized correctly. Could not infer"})

    device = kwargs.get("device", -1)
    if device == -1 and torch.cuda.is_available():
        logging.warning('CUDA available. To switch to GPU device pass `"parameters": {"device" : 0}` in the input.')
    if device == 0 and not torch.cuda.is_available():
        device = -1
        logging.warning("CUDA unavailable. Defaulting to CPU device.")

    device = "cuda" if device == 0 else "cpu"

    logging.info(f"Using device: {device} for the inference")

    try:
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        print(type(data))
        print(data)
        # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").input_features.to(device)
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        # model = model.to(device)
        preds = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
        result = tokenizer.batch_decode(preds, skip_special_tokens=True)[0]
        return json.dumps({
            "result": result
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    print(init())
    print(run("data"))
