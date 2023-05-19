import argparse
import pandas as pd
from datasets import load_dataset
import json

test_data = "./small-test.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--test-data", type=str, default=test_data, dest="test_data_file")
args = parser.parse_args()

df = pd.DataFrame(
    load_dataset("glue", "mnli", split="validation_matched", streaming=True).take(1000)
)

id2label = {
    "id2label": {"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"},
    "label2id": {"ENTAILMENT": 0, "CONTRADICTION": 2, "NEUTRAL": 1},
}
id2label = id2label["id2label"]
label_df = pd.DataFrame.from_dict(id2label, orient="index", columns=["label_string"])
label_df["label"] = label_df.index.astype("int64")
label_df = label_df[["label", "label_string"]]

# join the train, validation and test dataframes with the id2label dataframe to get the label_string column
df = df.merge(label_df, on="label", how="left")
# concat the premise and hypothesis columns to with "[CLS]" in the beginning and "[SEP]" in the middle and end to get the text column
df["input_string"] = "[CLS] " + df["premise"] + " [SEP] " + df["hypothesis"] + " [SEP]"
# drop the idx, premise and hypothesis columns as they are not needed
df = df.drop(columns=["idx", "premise", "hypothesis"])

df.to_json(args.test_data_file, lines=True, orient="records")
