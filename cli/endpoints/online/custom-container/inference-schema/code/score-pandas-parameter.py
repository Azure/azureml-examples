from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.schema_decorators import input_schema, output_schema
from transformers import BertForMaskedLM, FillMaskPipeline, AutoTokenizer
from operator import itemgetter
from pathlib import Path
import pandas as pd
import numpy as np
import os


def init():
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR")).resolve()
    model_dir = model_dir / "models/bert-base-uncased"
    model = BertForMaskedLM.from_pretrained(
        model_dir, config=(model_dir / "config.json"), local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        config=(model_dir / "tokenizer_config.json"),
        padding=True,
        return_tensors="pt",
    )
    global pipe
    pipe = FillMaskPipeline(model=model, tokenizer=tokenizer)


# Inference Schema ParameterType objects are defined using sample objects
param_pandas_df = PandasParameterType(
    pd.DataFrame(
        [
            {"sentence": "Hello my dog is [MASK] and I love her"},
            {"sentence": "What [MASK] is the show?"},
        ]
    )
)


@input_schema(param_name="pandas_df", param_type=param_pandas_df)
def run(pandas_df):
    # JSON deserialization is handled automatically
    masked_sentences = pandas_df["sentence"].tolist()
    results = pipe(masked_sentences)
    results = (results,) if len(masked_sentences) == 1 else results
    results = (sorted(r, key=itemgetter("score"), reverse=True) for r in results)
    results = [[token["token_str"] for token in result] for result in results]
    return results
