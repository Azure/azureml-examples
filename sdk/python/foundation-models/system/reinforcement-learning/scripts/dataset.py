import os
import json
import subprocess
import pandas as pd
from tempfile import TemporaryDirectory
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from typing import Optional
from json import JSONDecodeError
import requests
from tqdm import tqdm


SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def register_dataset(ml_client: MLClient, dataset_name: str, file_path: str):
    """Register a dataset in Azure ML."""
    data_asset = Data(
        name=dataset_name,
        path=file_path,
        type=AssetTypes.URI_FILE,
        description="FinQA dataset",
        tags={"source": "https://github.com/czyssrs/FinQA"},
        version="1",
    )

    registered_data = ml_client.data.create_or_update(data_asset)
    print(f"Registered dataset {registered_data.name}.")
    return registered_data


def download_finqa_dataset(src: str, target_dir: str = "data/raw"):
    """Prepare the FinQA dataset for training and evaluation."""
    with TemporaryDirectory() as tmpdir:
        print(f"Cloning raw FinQA dataset to {tmpdir} ...")
        subprocess.run(["git", "clone", src, tmpdir], check=True)
        os.makedirs(target_dir, exist_ok=True)
        print("Converting FinQA dataset to jsonl format ...")
        dataset_dir = os.path.join(tmpdir, "dataset")
        filenames = ["train.json", "dev.json", "test.json"]
        for filename in filenames:
            target_file_name = filename.split(".")[0] + ".jsonl"
            convert_to_jsonl(
                current_path=os.path.join(dataset_dir, filename),
                target_path=os.path.join(target_dir, target_file_name),
            )


def convert_to_jsonl(current_path: str, target_path: str):
    """Convert FinQA dataset file to jsonl format."""
    with open(current_path, "r") as rf, open(target_path, "w") as wf:
        lines = json.loads(rf.read())
        for item in lines:
            wf.write(json.dumps(item) + "\n")
    print(f"Converted {current_path} to {target_path}.")


def prepare_finqa_dataset(
    ml_client: MLClient, data_dir: str = "data", register_datasets: bool = False
) -> tuple[str, str, str]:
    """Prepare the FinQA dataset for training and evaluation."""
    # VERL finetuning relies on acceptable data sources for reward modeling and evaluation
    data_source = "openai/gsm8k"

    # download and convert dataset
    raw_data_dir = os.path.join(data_dir, "raw")
    FINQA_GIT_REPO = "https://github.com/czyssrs/FinQA"
    download_finqa_dataset(src=FINQA_GIT_REPO, target_dir=raw_data_dir)
    train_dataset_path = os.path.join(raw_data_dir, "train.jsonl")
    test_dataset_path = os.path.join(raw_data_dir, "test.jsonl")
    valid_dataset_path = os.path.join(raw_data_dir, "dev.jsonl")

    def format_list_to_string(data_list: list):
        """Convert list to string with newline separation"""
        if not data_list:
            return ""
        if isinstance(data_list, str):
            return data_list
        return "\n".join(str(item) for item in data_list)

    def format_table(table_list: list):
        """Format table data as string"""
        if not table_list:
            return ""
        table_str = "\nTable:\n"
        for row in table_list:
            if isinstance(row, list):
                table_str += " | ".join(str(cell) for cell in row) + "\n"
            else:
                table_str += str(row) + "\n"
        return table_str

    def map_fn(example: pd.Series, idx: int, split: str):
        """Map function to transform each example into desired format."""
        pre_instruction = "Please answer the following financial question based on the context provided."
        post_instruction = (
            'Let\'s think step by step and output the final answer after "####".'
        )
        qa = example.get("qa", {})
        question = qa.get("question", "")
        answer = qa.get("answer", qa.get("exe_ans", ""))
        gold_evidence = "\n".join(qa.get("gold_inds", {}).values())
        pre_text = format_list_to_string(example.get("pre_text", []))
        post_text = format_list_to_string(example.get("post_text", []))
        table = format_table(example.get("table", [])).strip()
        # Build prompt content according to specified schema
        prompt_content = "\n\n".join(
            [
                pre_instruction,
                "Context: " + pre_text,
                gold_evidence,
                post_text,
                table,
                "Question: " + question,
                post_instruction,
            ]
        )
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            "ability": "financial_reasoning",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "index": idx,
                "answer": answer,
                "question": question,
                "split": split,
            },
        }
        return data

    # load datasets
    train_dataset = pd.read_json(train_dataset_path, lines=True)
    test_dataset = pd.read_json(test_dataset_path, lines=True)
    valid_dataset = pd.read_json(valid_dataset_path, lines=True)

    # map datasets
    train_dataset = train_dataset.apply(
        lambda x: map_fn(x, x.name, split="train"), axis=1
    )
    test_dataset = test_dataset.apply(lambda x: map_fn(x, x.name, split="test"), axis=1)
    valid_dataset = valid_dataset.apply(
        lambda x: map_fn(x, x.name, split="valid"), axis=1
    )

    # save locally as jsonl
    train_dataset_path = os.path.join(data_dir, "train.jsonl")
    test_dataset_path = os.path.join(data_dir, "test.jsonl")
    valid_dataset_path = os.path.join(data_dir, "valid.jsonl")
    train_dataset.to_json(train_dataset_path, orient="records", lines=True)
    test_dataset.to_json(test_dataset_path, orient="records", lines=True)
    valid_dataset.to_json(valid_dataset_path, orient="records", lines=True)

    # register datasets
    if register_datasets:
        train_data = register_dataset(ml_client, "finqa_train", train_dataset_path)
        test_data = register_dataset(ml_client, "finqa_test", test_dataset_path)
        valid_data = register_dataset(ml_client, "finqa_valid", valid_dataset_path)
        if (
            (train_data and train_data.id)
            and (test_data and test_data.id)
            and (valid_data and valid_data.id)
        ):
            return train_data.id, test_data.id, valid_data.id

    return train_dataset_path, test_dataset_path, valid_dataset_path


def _is_file_valid_json(path):
    if not os.path.isfile(path):
        return False

    try:
        with open(path) as f:
            json.load(f)
        return True
    except JSONDecodeError as e:
        print(
            f"{path} exists but json loading fails ({e=}), thus treat as invalid file"
        )
        return False


def _download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if _is_file_valid_json(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def prepare_sharegpt_dataset(dataset_path="./data/draft_model/sharegpt_train_processed.jsonl") -> str:
    """Prepare the ShareGPT dataset for training the draft model."""
    # Download sharegpt if necessary
    if not os.path.isfile(dataset_path):
        temp_dataset_path = _download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(temp_dataset_path) as f:
        temp_dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    temp_dataset = [data for data in temp_dataset if len(data["conversations"]) >= 2]

    # Keep one conversation in one list
    new_dataset = []
    for temp_data in temp_dataset:
        if len(temp_data["conversations"]) % 2 != 0:
            continue
        if temp_data["conversations"][0]["from"] != "human":
            continue

        new_conversations = []

        for i in range(0, len(temp_data["conversations"]), 2):
            new_conversations.extend([
                {
                    "role": "user",
                    "content": temp_data["conversations"][i]["value"],
                },
                {
                    "role": "assistant",
                    "content": temp_data["conversations"][i + 1]["value"],
                }
            ])
        
        new_data = {}
        new_data["id"] = temp_data.get("id", "")
        new_data["conversations"] = new_conversations

        new_dataset.append(new_data)
    
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w") as f:
        for item in new_dataset:
            f.write(json.dumps(item) + "\n")
    
    return dataset_path
