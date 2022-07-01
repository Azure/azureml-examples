import sklearn as sk
import sklearn.datasets 
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import json
import joblib
import argparse

def train(sample_input, model_pkl):
      iris = sk.datasets.load_iris()
      X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=.9)
      json.dump({"data" : X_test.tolist()}, open(sample_input, "w"))
      model = Pipeline(steps=[("scaler", StandardScaler()), 
                              ("clf", RandomForestClassifier())])
      model.fit(X_train, y_train)
      print("F1 score: %s" % f1_score(y_test, model.predict(X_test), average="weighted"))
      joblib.dump(model, filename=model_pkl)

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("--sample-input", required=False, default="./endpoints/online/inference-schema/sample-input-iris.json")
      parser.add_argument("--model-pkl", required=False, default="./endpoints/online/inference-schema/iris.pkl")
      args = parser.parse_args()
      train(args.sample_input, args.model_pkl)