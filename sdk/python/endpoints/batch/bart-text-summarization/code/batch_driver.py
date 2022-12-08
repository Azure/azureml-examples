import os
import numpy as np
from transformers import pipeline, AutoTokenizer, TFBartForConditionalGeneration
from datasets import load_dataset


def init():
    global model
    global tokenizer

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, truncation=True, max_length=1024
    )
    model = TFBartForConditionalGeneration.from_pretrained(model_path)


def run(mini_batch):
    resultList = []

    ds = load_dataset("csv", data_files={"score": mini_batch})
    for text in ds["score"]["text"]:
        # perform inference
        input_ids = tokenizer.batch_encode_plus(
            [text], truncation=True, padding=True, max_length=1024
        )["input_ids"]
        summary_ids = model.generate(
            input_ids, max_length=130, min_length=30, do_sample=False
        )
        summaries = [
            tokenizer.decode(
                s, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for s in summary_ids
        ]

        # Get results:
        resultList.append(summaries[0])

    return resultList
