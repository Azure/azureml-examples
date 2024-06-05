# import library to parse command line arguments
import argparse, os

parser = argparse.ArgumentParser()
# add an argument to specify a dataset name to download
parser.add_argument("--dataset", type=str, default="wmt16", help="dataset name")
# add an argument to specify a dataset name to download
parser.add_argument(
    "--dataset_subset", type=str, default="ro-en", help="dataset subset name"
)
# argument to save a fraction of the dataset
parser.add_argument(
    "--fraction", type=float, default=1, help="fraction of the dataset to save"
)
# add an argument to specify the directory to download the dataset to
parser.add_argument(
    "--download_dir",
    type=str,
    default="wmt16-en-ro-dataset",
    help="directory to download the dataset to",
)
args = parser.parse_args()

# create the download directory if it does not exist
if not os.path.exists(args.download_dir):
    os.makedirs(args.download_dir)


def format_translation(example):
    for key in example["translation"]:
        example[key] = example["translation"][key]
    return example


# import hugging face datasets library
from datasets import load_dataset, get_dataset_split_names

for split in get_dataset_split_names(args.dataset, args.dataset_subset):
    # load the split of the dataset
    dataset = load_dataset(args.dataset, args.dataset_subset, split=split)
    dataset = dataset.map(format_translation, remove_columns=["translation"])
    # save the split of the dataset to the download directory as json lines file
    dataset.to_json(os.path.join(args.download_dir, f"{split}.jsonl"))

# import pandas library
import pandas as pd

# load the train.jsonl, test.jsonl and validation.jsonl files from the ./wmt16-en-ro-dataset/ folder and show first 5 rows
train_df = pd.read_json(os.path.join(args.download_dir, "train.jsonl"), lines=True)
validation_df = pd.read_json(
    os.path.join(args.download_dir, "validation.jsonl"), lines=True
)
test_df = pd.read_json(os.path.join(args.download_dir, "test.jsonl"), lines=True)

# change the frac parameter to control the number of examples to be saved
# save a fraction of the rows from the validation and test dataframes into files with small_ prefix in the ./wmt16-en-ro-dataset folder
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

# read ./wmt16-en-ro-dataset/small_test.jsonl into a pandas dataframe
import json

test_df = pd.read_json(
    os.path.join(args.download_dir, "small_test.jsonl"), orient="records", lines=True
)
# take 1 random sample
test_df = test_df.sample(n=1)
# rebuild index
test_df.reset_index(drop=True, inplace=True)

# create a json object with the key as "inputs" and value as a list of values from the en column of the test dataframe
test_df_copy = test_df[["en"]]
test_json = {"input_data": test_df_copy.to_dict("split")}
# save the json object to a file named sample_score.json in the ./wmt16-en-ro-dataset folder
with open(os.path.join(args.download_dir, "sample_score.json"), "w") as f:
    json.dump(test_json, f)
