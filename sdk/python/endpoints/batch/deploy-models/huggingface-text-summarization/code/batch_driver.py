import os
import time
import torch
import subprocess
import mlflow
from pprint import pprint
from transformers import AutoTokenizer, BartForConditionalGeneration
from optimum.bettertransformer import BetterTransformer
from datasets import load_dataset


def init():
    global model
    global tokenizer
    global device

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    if cuda_available:
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] ID of current CUDA device: {torch.cuda.current_device()}")
        print("[INFO] nvidia-smi output:")
        pprint(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode(
                "utf-8"
            )
        )
    else:
        print(
            "[WARN] CUDA acceleration is not available. This model takes hours to run on medium size data."
        )

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, truncation=True, max_length=1024
    )

    # Load the model
    try:
        model = BartForConditionalGeneration.from_pretrained(
            model_path, device_map="auto"
        )
    except Exception as e:
        print(
            f"[ERROR] Error happened when loading the model on GPU or the default device. Error: {e}"
        )
        print("[INFO] Trying on CPU.")
        model = BartForConditionalGeneration.from_pretrained(model_path)
        device = "cpu"

    # Optimize the model
    if device != "cpu":
        try:
            model = BetterTransformer.transform(model, keep_original_model=False)
            print("[INFO] BetterTransformer loaded.")
        except Exception as e:
            print(
                f"[ERROR] Error when converting to BetterTransformer. An unoptimized version of the model will be used.\n\t> {e}"
            )

    mlflow.log_param("device", device)
    mlflow.log_param("model", type(model).__name__)


def run(mini_batch):
    resultList = []

    print(f"[INFO] Reading new mini-batch of {len(mini_batch)} file(s).")
    ds = load_dataset("csv", data_files={"score": mini_batch})

    start_time = time.perf_counter()
    for idx, text in enumerate(ds["score"]["text"]):
        # perform inference
        inputs = tokenizer.batch_encode_plus(
            [text], truncation=True, padding=True, max_length=1024, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        summary_ids = model.generate(
            input_ids, max_length=130, min_length=30, do_sample=False
        )
        summaries = tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Get results:
        resultList.append(summaries[0])
        rps = idx / (time.perf_counter() - start_time + 00000.1)
        print("Rows per second:", rps)

    mlflow.log_metric("rows_per_second", rps)
    return resultList
