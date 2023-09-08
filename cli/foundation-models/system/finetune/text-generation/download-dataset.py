# import library to parse command line arguments
import argparse, os
import json

parser = argparse.ArgumentParser()
# add an argument to specify a dataset name to download
parser.add_argument("--dataset", type=str, default="samsum", help="dataset name")
# add an argument to specify a dataset name to download
parser.add_argument(
    "--dataset_subset", type=str, default="split", help="dataset subset name"
)
# add an argument to specify the directory to download the dataset to
parser.add_argument(
    "--download_dir",
    type=str,
    default="data",
    help="directory to download the dataset to",
)
args = parser.parse_args()

# create the download directory if it does not exist
if not os.path.exists(args.download_dir):
    os.makedirs(args.download_dir)


# import hugging face datasets library
from datasets import load_dataset, get_dataset_split_names

for split in get_dataset_split_names(args.dataset):
    # load the split of the dataset
    dataset = load_dataset(args.dataset, split=split)
    # save the split of the dataset to the download directory as json lines file
    dataset.to_json(os.path.join(args.download_dir, f"{split}.jsonl"))
    # print dataset features


# preprocess the dataset
import pandas as pd


def get_preprocessed_samsum(df):
    prompt = f"Summarize this dialog:\n{{}}\n---\nSummary:\n"

    df["text"] = df["dialogue"].map(prompt.format)
    df = df.drop(columns=["dialogue", "id"])
    df = df[["text", "summary"]]

    return df


test_df = pd.read_json("./samsum-dataset/test.jsonl", lines=True)
train_df = pd.read_json("./samsum-dataset/train.jsonl", lines=True)
validation_df = pd.read_json("./samsum-dataset/validation.jsonl", lines=True)
# map the train, validation and test dataframes to preprocess function
train_df = get_preprocessed_samsum(train_df)
validation_df = get_preprocessed_samsum(validation_df)
test_df = get_preprocessed_samsum(test_df)

# Save the preprocessed data
frac = 1
train_df.sample(frac=frac).to_json(
    "./samsum-dataset/small_train.jsonl", orient="records", lines=True
)
validation_df.sample(frac=frac).to_json(
    "./samsum-dataset/small_validation.jsonl", orient="records", lines=True
)
test_df.sample(frac=frac).to_json(
    "./samsum-dataset/small_test.jsonl", orient="records", lines=True
)


# read ./samsum-dataset/small_test.jsonl into a pandas dataframe
test_df = pd.read_json("./samsum-dataset/small_test.jsonl", lines=True)
# take 2 random samples
test_df = test_df.sample(n=2)
# rebuild index
test_df.reset_index(drop=True, inplace=True)
test_df.head(2)

# create a json object with the key as "input_data" and value as a list of values from the text column of the test dataframe
test_json = {"input_data": {"text": list(test_df["text"])}}
# save the json object to a file named sample_score.json in the ./samsum-dataset folder
with open("./samsum-dataset/sample_score.json", "w") as f:
    json.dump(test_json, f)
