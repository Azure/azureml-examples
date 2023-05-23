# import library to parse command line arguments
import argparse, os

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
    default="data",
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
    print(dataset.features)
    # save the split of the dataset to the download directory as json lines file
    dataset.select(range(int(dataset.num_rows * args.fraction))).to_json(
        os.path.join(args.download_dir, f"{split}.jsonl")
    )
