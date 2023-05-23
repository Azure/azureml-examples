import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pickle
import shutil

parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--model_output", type=str, help="Path of output model")
parser.add_argument("--run_id_output", type=str, help="runid_output")

args = parser.parse_args()

training_df = pd.read_parquet(os.path.join(args.training_data, "data"))

categorical_feature_names = ["transactionID", "accountID"]
ordinal_feature_names = [
    "isProxyIP",
    # this feature is from feature store "isUserRegistered",
]

X = (
    training_df.drop(
        categorical_feature_names + ordinal_feature_names + ["is_fraud", "timestamp"],
        axis="columns",
    )
    .fillna(0)
    .to_numpy()
)
y = training_df["is_fraud"].astype(int).to_numpy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
clf = RandomForestClassifier(
    n_estimators=11,
    random_state=42,
)

# train the model
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
y_pred

y_prob = clf.predict_proba(X_test)
y_prob

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(
    f"""Precision: {precision},
Recall: {recall},
F1: {f1}"""
)

confusion_matrix(y_test, y_pred)

# save the model
pkl_filename = os.path.join(args.model_output, "clf.pkl")
with open(pkl_filename, "wb") as file:
    pickle.dump(clf, file)

# save the feature_retrieval_spec
shutil.copy(
    os.path.join(args.training_data, "feature_retrieval_spec.yaml"), args.model_output
)

# write runid to output file

env_runid = os.environ.get("AZUREML_RUN_ID")
f = open(args.run_id_output, "x")
f.write(env_runid)
f.close()
