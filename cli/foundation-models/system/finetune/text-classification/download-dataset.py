# import library to parse command line arguments
import argparse, os

parser = argparse.ArgumentParser()
# add an argument to specify a dataset name to download
parser.add_argument(
    "--dataset", type=str, default="dair-ai/emotion", help="dataset name"
)
# add an argument to specify a dataset name to download
parser.add_argument(
    "--dataset_subset", type=str, default="split", help="dataset subset name"
)
# add an argument to specify the directory to download the dataset to
parser.add_argument(
    "--download_dir",
    type=str,
    default="emotion-dataset",
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


# import hugging face datasets library
from datasets import load_dataset, get_dataset_split_names
from functools import partial

for split in get_dataset_split_names(args.dataset):
    # load the split of the dataset
    dataset = load_dataset(args.dataset, split=split)
    # save the split of the dataset to the download directory as json lines file
    dataset.to_json(os.path.join(args.download_dir, f"{split}.jsonl"))
    # print dataset features

# get label2id and id2label mapping

# get any split of data
split = get_dataset_split_names(args.dataset)[0]
dataset = load_dataset(args.dataset, split=split)

labels = dataset.features["label"].names

id2label = {}
label2id = {}

for i, label in enumerate(labels):
    id2label[i] = label
    label2id[label] = i

label_mapping = {"id2label": id2label, "label2id": label2id}

import json

with open(os.path.join(args.download_dir, "label.json"), "w") as f:
    json.dump(label_mapping, f)

# load the id2label json element of the ./emotion-dataset/label.json file into pandas table with keys as 'label' column of int64 type and values as 'label_string' column as string type
import json
import pandas as pd

with open(os.path.join(args.download_dir, "label.json")) as f:
    id2label = json.load(f)
    id2label = id2label["id2label"]
    label_df = pd.DataFrame.from_dict(
        id2label, orient="index", columns=["label_string"]
    )
    label_df["label"] = label_df.index.astype("int64")
    label_df = label_df[["label", "label_string"]]

test_df = pd.read_json(os.path.join(args.download_dir, "test.jsonl"), lines=True)
train_df = pd.read_json(os.path.join(args.download_dir, "train.jsonl"), lines=True)
validation_df = pd.read_json(
    os.path.join(args.download_dir, "validation.jsonl"), lines=True
)
# join the train, validation and test dataframes with the id2label dataframe to get the label_string column
train_df = train_df.merge(label_df, on="label", how="left")
validation_df = validation_df.merge(label_df, on="label", how="left")
test_df = test_df.merge(label_df, on="label", how="left")

# save a fraction of the rows from the validation and test dataframes into files with small_ prefix in the ./emotion-dataset folder
train_df.sample(frac=args.fraction).to_json(
    "./emotion-dataset/small_train.jsonl", orient="records", lines=True
)
validation_df.sample(frac=args.fraction).to_json(
    "./emotion-dataset/small_validation.jsonl", orient="records", lines=True
)
test_df.sample(frac=args.fraction).to_json(
    "./emotion-dataset/small_test.jsonl", orient="records", lines=True
)

# read ./emotion-dataset/small_test.jsonl into a pandas dataframe
test_df = pd.read_json(os.path.join(args.download_dir, "small_test.jsonl"), lines=True)
# take 10 random samples
test_df = test_df.sample(n=10)
# rebuild index
test_df.reset_index(drop=True, inplace=True)
# rename the label_string column to ground_truth_label
test_df = test_df.rename(columns={"label_string": "ground_truth_label"})

# create a json object with the key as "inputs" and value as a list of values from the text column of the test dataframe
test_df_copy = test_df[["text"]]
test_json = {"input_data": test_df_copy.to_dict("split")}
# save the json object to a file named sample_score.json in the ./emotion-dataset folder
with open(os.path.join(args.download_dir, "sample_score.json"), "w") as f:
    json.dump(test_json, f)
