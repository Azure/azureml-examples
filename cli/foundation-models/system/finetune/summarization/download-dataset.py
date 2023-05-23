# import library to parse command line arguments
import argparse, os
import pandas as pd
import os

parser = argparse.ArgumentParser()
# add an argument to specify a dataset name to download
parser.add_argument("--dataset", type=str, default="cnn_dailymail", help="dataset name")
# add an argument to specify the config name of the dataset
parser.add_argument(
    "--config_name", type=str, default="3.0.0", help="config name of the dataset"
)
# argument to save a fraction of the dataset
parser.add_argument(
    "--fraction", type=float, default=1, help="fraction of the dataset to save"
)
# add an argument to specify the directory to download the dataset to
parser.add_argument(
    "--download_dir",
    type=str,
    default="./news-summary-dataset",
    help="directory to download the dataset to",
)
args = parser.parse_args()

# create the download directory if it does not exist
if not os.path.exists(args.download_dir):
    os.makedirs(args.download_dir)

# import hugging face datasets library
from datasets import load_dataset, get_dataset_split_names

for split in get_dataset_split_names(args.dataset, config_name=args.config_name):
    print(f"Loading {split} split of {args.dataset} dataset...")
    # load the split of the dataset
    dataset = load_dataset(args.dataset, args.config_name, split=split)
    # save the split of the dataset to the download directory as json lines file
    dataset.to_json(os.path.join(args.download_dir, f"{split}.jsonl"))

train_df = pd.read_json(os.path.join(args.download_dir, "train.jsonl"), lines=True)
validation_df = pd.read_json(
    os.path.join(args.download_dir, "validation.jsonl"), lines=True
)
test_df = pd.read_json(os.path.join(args.download_dir, "test.jsonl"), lines=True)

# drop the id column as it is not needed for fine tuning
train_df.drop(columns=["id"], inplace=True)
validation_df.drop(columns=["id"], inplace=True)
test_df.drop(columns=["id"], inplace=True)

# save a fraction of the rows from the validation and test dataframes into files with small_ prefix in the ./news-summary-dataset folder
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

# generate sample scoring data
# read ./news-summary-dataset/small_test.jsonl into a pandas dataframe
import pandas as pd
import json

test_df = pd.read_json(
    os.path.join(args.download_dir, "small_test.jsonl"), orient="records", lines=True
)
# take 1 random sample
test_df = test_df.sample(n=1)
# rebuild index
test_df.reset_index(drop=True, inplace=True)
# rename the highlights column to ground_truth_summary
test_df.rename(columns={"highlights": "ground_truth_summary"}, inplace=True)
# create a json object with the key as "inputs" and value as a list of values from the article column of the test dataframe
test_df_copy = test_df[["article"]]
test_json = {"input_data": test_df_copy.to_dict("split")}
# save the json object to a file named sample_score.json in the ./emotion-dataset folder
with open(os.path.join(args.download_dir, "sample_score.json"), "w") as f:
    json.dump(test_json, f)
