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
    default="data",
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
    dataset.select(range(int(dataset.num_rows * args.fraction))).to_json(
        os.path.join(args.download_dir, f"{split}.jsonl")
    )
