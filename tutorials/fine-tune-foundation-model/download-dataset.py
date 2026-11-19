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
    default="data",
    help="directory to download the dataset to",
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
