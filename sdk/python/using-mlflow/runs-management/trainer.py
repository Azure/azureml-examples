import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def train_and_log(input_data: str, register_model: bool = False):
    # Models are not logged since we want to log a pipeline
    mlflow.xgboost.autolog(log_models=False)

    model_name = "heart-classifier"

    df = pd.read_csv(input_data)

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1), df["target"], test_size=0.3
    )

    encoder = ColumnTransformer(
        [
            (
                "cat_encoding",
                OrdinalEncoder(
                    categories="auto",
                    encoded_missing_value=np.nan,
                ),
                ["thal"],
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    pipeline = Pipeline(steps=[("encoding", encoder), ("model", model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    mlflow.log_metric("accuracy", accuracy)

    # Log the model with its signature
    signature = infer_signature(X_test, y_test)
    if register_model:
        mlflow.sklearn.log_model(pipeline, artifact_path="pipeline", registered_model_name=model_name, signature=signature)
    else:
        mlflow.sklearn.log_model(pipeline, artifact_path="pipeline", signature=signature)