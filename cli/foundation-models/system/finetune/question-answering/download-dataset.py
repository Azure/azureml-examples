# import library to parse command line arguments
import argparse, os

parser = argparse.ArgumentParser()
# add an argument to specify a dataset name to download
parser.add_argument("--dataset", type=str, default="squad", help="dataset name")
# add an argument to specify the directory to download the dataset to
parser.add_argument(
    "--download_dir",
    type=str,
    default="./squad-dataset",
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

for split in get_dataset_split_names(args.dataset):
    # load the split of the dataset
    dataset = load_dataset(args.dataset, split=split)
    # save the split of the dataset to the download directory as json lines file
    dataset.to_json(os.path.join(args.download_dir, f"{split}.jsonl"))
    # print dataset features

# load the train.jsonl and validation.jsonl files from the ./squad-dataset/ folder and show first 5 rows
import pandas as pd

train_df = pd.read_json(os.path.join(args.download_dir, "train.jsonl"), lines=True)
validation_df = pd.read_json(
    os.path.join(args.download_dir, "validation.jsonl"), lines=True
)

# save a fraction of the rows from the train dataframe into files with small_ prefix in the ./squad-dataset folder
train_df.sample(frac=args.fraction).to_json(
    os.path.join(args.download_dir, "small_train.jsonl"), orient="records", lines=True
)
# the original dataset does not have a test split, so split the validation dataframe into validation and test dataframes equally
validation_df, test_df = (
    validation_df[: len(validation_df) // 2],
    validation_df[len(validation_df) // 2 :],
)
# save a fraction of the rows from the validation and test dataframes into files with small_ prefix in the ./squad-dataset folder
validation_df.sample(frac=args.fraction).to_json(
    os.path.join(args.download_dir, "small_validation.jsonl"),
    orient="records",
    lines=True,
)
test_df.sample(frac=args.fraction).to_json(
    os.path.join(args.download_dir, "small_test.jsonl"), orient="records", lines=True
)

# read ./squad-dataset/small_test.jsonl into a pandas dataframe
import json

test_df = pd.read_json("./squad-dataset/small_test.jsonl", orient="records", lines=True)
# take 10 random samples
test_df = test_df.sample(n=10)
# rebuild index
test_df.reset_index(drop=True, inplace=True)
# flatten the json object in the "answer" column with the keys "answer_start" and "text"
json_struct = json.loads(test_df.to_json(orient="records"))
test_df = pd.json_normalize(json_struct)
# drop id and title columns
test_df = test_df.drop(columns=["id", "title"])

# create a json object with "inputs" as key and a list of json objects with "question" and "context" as keys
test_df_copy = test_df[["question", "context"]]
test_json = {"input_data": test_df_copy.to_dict("split")}

# write the json object to a file named sample_score.json in the ./squad-dataset folder
with open("./squad-dataset/sample_score.json", "w") as f:
    json.dump(test_json, f)
