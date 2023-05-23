# import library to parse command line arguments
import argparse, os

parser = argparse.ArgumentParser()
# add an argument to specify a dataset name to download
parser.add_argument("--dataset", type=str, default="conll2003", help="dataset name")
# add an argument to specify the directory to download the dataset to
parser.add_argument(
    "--download_dir",
    type=str,
    default="conll2003-dataset",
    help="directory to download the dataset to",
)
# argument to save a fraction of the dataset
parser.add_argument(
    "--fraction", type=float, default=1, help="fraction of the dataset to save"
)
args = parser.parse_args()

# create the download directory if it does not exist
if not os.path.exists(args.download_dir):
    os.makedirs(args.download_dir)


def format_ner_tags(example, class_names):
    example["text"] = " ".join(example["tokens"])
    example["ner_tags_str"] = [class_names[id] for id in example["ner_tags"]]
    return example


# import hugging face datasets library
from datasets import load_dataset, get_dataset_split_names
from functools import partial

for split in get_dataset_split_names(args.dataset):
    # load the split of the dataset
    dataset = load_dataset(args.dataset, split=split)
    dataset = dataset.map(
        partial(format_ner_tags, class_names=dataset.features["ner_tags"].feature.names)
    )
    # save the split of the dataset to the download directory as json lines file
    dataset.to_json(os.path.join(args.download_dir, f"{split}.jsonl"))
    # print dataset features

import pandas as pd

# load test.jsonl, train.jsonl and validation.jsonl form the ./conll2003-dataset folder into pandas dataframes
test_df = pd.read_json(os.path.join(args.download_dir, "test.jsonl"), lines=True)
train_df = pd.read_json(os.path.join(args.download_dir, "train.jsonl"), lines=True)
validation_df = pd.read_json(
    os.path.join(args.download_dir, "validation.jsonl"), lines=True
)

# save a fraction of the rows from the validation and test dataframes into files with small_ prefix in the ./conll2003-dataset folder
train_df.sample(frac=args.fraction).to_json(
    os.path.join(args.download_dir, "small_train.jsonl"), orient="records", lines=True
)
validation_df.sample(frac=args.fraction).to_json(
    os.path.join(args.download_dir, "small_validation.jsonl"),
    orient="records",
    lines=True,
)
test_df.sample(frac=args.fraction).to_json(
    os.path.join(args.download_dir, "small_test.jsonl"), orient="records", lines=True
)


# read ./conll2003-dataset/small_test.jsonl into a pandas dataframe
test_df = pd.read_json(os.path.join(args.download_dir, "small_test.jsonl"), lines=True)
# take 10 random samples
test_df = test_df.sample(n=10)
# drop the id, pos_tags, chunk_tags, ner_tags column
test_df.drop(columns=["id", "pos_tags", "chunk_tags", "ner_tags"], inplace=True)
# rebuild index
test_df.reset_index(drop=True, inplace=True)
# rename the ner_tags_str column to ground_truth_label
test_df = test_df.rename(columns={"ner_tags_str": "ground_truth_tags"})

import json

# create a json object with the key as "inputs" and value as a list of values from the text column of the test dataframe
test_df_copy = test_df[["tokens"]]
test_json = {"input_data": test_df_copy.to_dict("split")}
# save the json object to a file named sample_score.json in the ./conll2003-dataset folder
with open(os.path.join(args.download_dir, "sample_score.json"), "w") as f:
    json.dump(test_json, f)
