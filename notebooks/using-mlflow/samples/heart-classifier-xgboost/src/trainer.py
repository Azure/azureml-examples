import mlflow
import pandas as pd
from mlflow.models import infer_signature
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
df = pd.read_csv(file_url)
df["thal"] = df["thal"].astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1), df["target"], test_size=0.3
)

artifact_path = "classifier"
model_name = "heart-classifier"

# log_models=False as a fix until TensorSpec supported
mlflow.xgboost.autolog(log_models=False)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

mlflow.log_metric("accuracy", accuracy)

# Log the model with its signature
signature = infer_signature(X_test, y_pred)
mlflow.xgboost.log_model(
    model, artifact_path, registered_model_name=model_name, signature=signature
)
