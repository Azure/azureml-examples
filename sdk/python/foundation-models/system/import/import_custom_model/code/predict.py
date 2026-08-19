# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model = None
tokenizer = None


def init():
    global model
    global tokenizer

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "INPUT_model_path")
    try:
        print(f"Loading model from path {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, device_map="auto", trust_remote_code=True
        )
        print("Loading successful.")
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_input_string(input_str):
    data = json.loads(input_str)
    input_data = data["inputs"]["input_str"]
    params = data["inputs"]["params"]
    return input_data, params


def run(data):
    global model
    global tokenizer

    if not model or not tokenizer:
        return json.dumps(
            {
                "error": "Model or tokenizer was not initialized correctly. Could not infer"
            }
        )

    print(f"input data:\n{data}")

    try:
        inputs, params = get_input_string(data)
    except Exception as e:
        return json.dumps(
            {
                "error": f"Error: {e}."
                + 'Input request should be in: {"inputs": {"input_str": ["text"], "params": {"k": "v"}}}'
            }
        )

    device = params.get("device", -1)
    if device == -1 and torch.cuda.is_available():
        print(
            'WARNING: CUDA available. To switch to GPU device pass `"params": {"device" : 0}` in the input.'
        )
    if device == 0 and not torch.cuda.is_available():
        device = -1
        print("WARNING: CUDA unavailable. Defaulting to CPU device.")

    device = "cuda" if device == 0 else "cpu"

    print(f"Using device: {device} for the inference")

    try:
        max_new_tokens = params.get("max_new_tokens", 512)
        result = []
        for inp in inputs:
            inputs = tokenizer(inp, return_tensors="pt").to(device)
            model = model.to(device)
            preds = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True
            )
            result.append(tokenizer.batch_decode(preds, skip_special_tokens=True)[0])
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    print(init())
    print(
        run(
            '{"inputs": {"input_str": ["rocco noticed the almost defeated look on her lovely face and did not like it.", "Hello, my dog is cute."], "params": {"max_new_tokens": 128, "device": -1}}}'
        )
    )
