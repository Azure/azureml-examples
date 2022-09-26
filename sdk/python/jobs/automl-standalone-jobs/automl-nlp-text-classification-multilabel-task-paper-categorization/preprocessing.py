import ast
import os

import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

N = 10  # the number of most popular labels to keep

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

# Create the folder if not already exists, save dataset
if not os.path.exists("data"):
    os.mkdir("data")
data.to_csv("./data/arxiv_abstract.csv", index=False)
