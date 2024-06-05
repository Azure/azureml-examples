import argparse
import pandas as pd
from datasets import load_dataset

test_data = "./small-test.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--test-data", type=str, default=test_data, dest="test_data_file")
args = parser.parse_args()

hf_test_data = load_dataset("conll2003", split="test", streaming=True)

test_data_df = pd.DataFrame(hf_test_data.take(1000))
# Picked from https://huggingface.co/datasets/conll2003
label_dict = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}
label_reverse_dict = {value: key for key, value in label_dict.items()}
test_data_df["input_string"] = test_data_df["tokens"].apply(lambda x: " ".join(x))
test_data_df["ner_tags_str"] = test_data_df["ner_tags"].apply(
    lambda x: str([label_reverse_dict[tag] for tag in x])
)

test_data_df.to_json(args.test_data_file, lines=True, orient="records")
