import os
import csv
import json
import random
import urllib
import argparse
import datasets
import pandas as pd

# Get the model name from argument
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
args = parser.parse_args()

# Define directories and filenames as variables
dataset_dir = "dataset"
test_datafile = "test_100.csv"

batch_dir = "batch"
batch_inputs_dir = os.path.join(batch_dir, "inputs")
batch_input_file = "batch_input.csv"
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(batch_dir, exist_ok=True)
os.makedirs(batch_inputs_dir, exist_ok=True)

testdata = datasets.load_dataset("bookcorpus", split="train", streaming=True)

test_df = pd.DataFrame(data=testdata.take(100))

# Get the right mask token from huggingface
with urllib.request.urlopen(
    f"https://huggingface.co/api/models/{args.model_name}"
) as url:
    data = json.load(url)
    mask_token = data["mask_token"]

# Take the value of the "text" column, replace a random word with the mask token, and save the result in the "masked_text" column
test_df["masked_text"] = test_df["text"].apply(
    lambda x: x.replace(random.choice(x.split()), mask_token, 1)
)

# Save the test_df dataframe to a csv file in the ./bookcorpus-dataset folder
test_df.to_csv(os.path.join(".", dataset_dir, test_datafile), index=False)

batch_df = test_df[["masked_text"]].rename(columns={"masked_text": "input_string"})

# Divide this into files of 10 rows each
batch_size_per_predict = 10
for i in range(0, len(batch_df), batch_size_per_predict):
    j = i + batch_size_per_predict
    batch_df[i:j].to_csv(
        os.path.join(batch_inputs_dir, str(i) + batch_input_file), quoting=csv.QUOTE_ALL
    )
