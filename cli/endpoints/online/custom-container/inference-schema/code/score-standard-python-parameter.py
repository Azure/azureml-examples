from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
from inference_schema.schema_decorators import input_schema, output_schema
from transformers import BertForMaskedLM, FillMaskPipeline, AutoTokenizer
from operator import itemgetter
from pathlib import Path
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
param_masked_sentences = StandardPythonParameterType(
    ["Hello my dog is [MASK] and I love her", "What [MASK] is the show?"]
)
param_top_k = StandardPythonParameterType(5)
param_tokens = StandardPythonParameterType(["time", "exactly", "good", "else", "year"])

# Inference Schema schema decorators are applied to the run function
# param_name corresponds to the named run function argument
# input_schema decorators can be stacked to specify multiple named run function arguments
@input_schema(param_name="masked_sentences", param_type=param_masked_sentences)
@input_schema(param_name="top_k", param_type=param_top_k)
@output_schema(param_tokens)
def run(masked_sentences, top_k):
    # JSON deserialization is handled automatically
    results = pipe(masked_sentences, top_k=top_k)
    results = (results,) if len(masked_sentences) == 1 else results
    results = (sorted(r, key=itemgetter("score"), reverse=True) for r in results)
    results = [[token["token_str"] for token in result] for result in results]
    return results
