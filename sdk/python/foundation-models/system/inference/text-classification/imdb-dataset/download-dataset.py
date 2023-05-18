# import library to parse command line arguments
import argparse, os, json

parser = argparse.ArgumentParser()
# add an argument to specify a dataset name to download
parser.add_argument("--dataset", type=str, default="imdb", help="dataset name")
# add an argument to specify the directory to download the dataset to
parser.add_argument(
    "--download_dir",
    type=str,
    default="./",
    help="directory to download the dataset to",
)
# add an argument to specify the split of the dataset to download
parser.add_argument(
    "--split", type=str, default="train", help="split of the dataset to download"
)
args = parser.parse_args()

# create the download directory if it does not exist
if not os.path.exists(args.download_dir):
    os.makedirs(args.download_dir)

# import hugging face datasets library
from datasets import load_dataset, get_dataset_split_names

print(f"Loading {args.split} split of {args.dataset} dataset...")
# load the split of the dataset
dataset = load_dataset(args.dataset, split=args.split, streaming=True)
dataset = dataset.shuffle()
dataset = dataset.take(5)
# save the split of the dataset to the download directory as json lines file
with open(os.path.join(args.download_dir, f"{args.split}.jsonl"), "w") as f:
    for line in dataset:
        f.write(json.dumps(line) + "\n")
