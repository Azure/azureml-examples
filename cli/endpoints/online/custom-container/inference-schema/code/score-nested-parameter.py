from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.schema_decorators import input_schema
from transformers import BertForMaskedLM, FillMaskPipeline, AutoTokenizer
from operator import itemgetter
from pathlib import Path
import pandas as pd
import numpy as np
import os
import joblib


def init():
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR")).resolve()
    bert_model_dir = model_dir / "models/bert-base-uncased"
    model = BertForMaskedLM.from_pretrained(
        bert_model_dir, config=(bert_model_dir / "config.json"), local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        bert_model_dir,
        config=(bert_model_dir / "tokenizer_config.json"),
        padding=True,
        return_tensors="pt",
    )

    global bert
    bert = FillMaskPipeline(model=model, tokenizer=tokenizer)

    global iris
    iris_model_dir = model_dir / "iris"
    iris = joblib.load(iris_model_dir / "iris.pkl")


param_input_k = NumpyParameterType(np.array([5]))
param_pandas_df = PandasParameterType(
    pd.DataFrame(
        [
            {"sentence": "Hello my dog is [MASK] and I love her"},
            {"sentence": "What [MASK] is the show?"},
        ]
    )
)

category_list = np.array(["Setosa", "Versicolor", "Virginica"])

param_input_iris = NumpyParameterType(np.random.random(4)[np.newaxis, ...])

param_input = StandardPythonParameterType(
    {"bert_input": param_pandas_df, "iris_input": param_input_iris}
)


def score_bert(sentences):
    results = bert(sentences)
    results = (results,) if len(sentences) == 1 else results
    results = (sorted(r, key=itemgetter("score"), reverse=True) for r in results)
    results = [[token["token_str"] for token in result] for result in results]
    return results


def score_iris(iris_input):
    probabilities = iris.predict_proba(iris_input)
    predicted_categories = np.argmax(probabilities, axis=1)
    predicted_categories = np.choose(predicted_categories, category_list).flatten()
    result = {
        "Probabilities": probabilities.tolist(),
        "Predicted Categories": predicted_categories.tolist(),
    }
    return result


@input_schema(param_name="input", param_type=param_input)
def run(input):
    results = {}
    # pandas_df is delivered as a pandas dataframe
    sentences = input["bert_input"]["sentence"].tolist()
    results["bert"] = score_bert(sentences)
    # iris_input is delivered as a numpy array
    results["iris"] = score_iris(input["iris_input"])
    return results
