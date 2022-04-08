import ast
import os

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

N = 10  # the number of most popular labels to keep
train_ratio = 0.2
valid_ratio = 0.4
test_ratio = 1 - train_ratio - valid_ratio
seed = 101

data = pd.read_csv("arxiv_data.csv")
data["terms"] = data["terms"].apply(ast.literal_eval)

transformer = MultiLabelBinarizer(sparse_output=True)
transformer.fit(data["terms"])
K = len(transformer.classes_)
print("The original dataset has {} unique labels".format(K))

counter = Counter()
for labels in data["terms"]:
    counter.update(labels)
min_count = counter.most_common(N)[-1]


def filter_labels(labels):
    labels = [label for label in labels if counter[label] >= 294]
    return labels


data["terms"] = data["terms"].apply(filter_labels)
data["titles"] = data["titles"].apply(lambda x: x.replace("\n", " "))
data["summaries"] = data["summaries"].apply(lambda x: x.replace("\n", " "))

all_index = np.arange(data.shape[0])
train_index, valid_index = train_test_split(all_index, train_size=train_ratio)
valid_index, test_index = train_test_split(
    valid_index, train_size=valid_ratio / (1 - train_ratio)
)

train_data = data.iloc[train_index, :]
valid_data = data.iloc[valid_index, :]
test_data = data.iloc[test_index, :]

# Create the folder if not already exists, save dataset
if not os.path.exists("data"):
    os.mkdir("data")
train_data.to_csv("./data/arxiv_abstract_train.csv", index=False)
valid_data.to_csv("./data/arxiv_abstract_valid.csv", index=False)
test_data.to_csv("./data/arxiv_abstract_test.csv", index=False)
