import argparse
from azureml.core.model import Model
from azureml.core import Run, Workspace
import os 

from importlib_metadata import requires
import numpy as np
import mlflow
import time
from typing import Any, List, Union, Dict, Callable

# from dataclasses import dataclass, field
from datasets import Metric, load_dataset, load_metric
from datasets import DatasetDict, Dataset # used for typing
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedTokenizerBase,
)

# Azure ML imports - could replace this with e.g. wandb or mlflow
from transformers.integrations import MLflowCallback


# Set the GLUE task
TASK = "mrpc"


# This line creates a handles to the current run. It is used for model registration
run = Run.get_context()


# Setup the tokenizer based on the mrpc task data format
def tokenize_sentence_pair(examples: Union[Dict, Any]) -> Union[Dict, Any]:
    return tokenizer(
        examples["sentence1"], examples["sentence2"], truncation=True
    )


# Load the dataset for the GLUE task, and apply the tokenizer        
def load_encoded_glue_dataset(
    task: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Union[DatasetDict, Dataset]:
    """Load GLUE data, apply tokenizer and split into train/validation."""
    raw_dataset = load_dataset("glue", task)
    encoded_dataset = raw_dataset.map(tokenize_sentence_pair, batched=True)

    return encoded_dataset["train"], encoded_dataset["validation"]

# Load the metrics for the GLUE task
metric = load_metric("glue", TASK)

# Setup the metric computation function
def compute_metrics_function(eval_pred: EvalPrediction) -> Dict:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":

    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument("--model_checkpoint", default="distilbert-base-uncased", required = False)
    parser.add_argument("--trained_model", type=str, default = "trained_model", help="path to model file", required = False)
    training_args, args = parser.parse_args_into_dataclasses()
    print("Training args", training_args)
    raw_dataset = load_dataset("glue", TASK)


    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    encoded_dataset = raw_dataset.map(tokenize_sentence_pair, batched=True)
    encoded_dataset_train, encoded_dataset_eval = encoded_dataset["train"], encoded_dataset["validation"]

    # Load the model from checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, num_labels=2
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_function,
    )

    trainer.pop_callback(MLflowCallback)

    print("Training...")

    start = time.time()
    trainer.train()
    mlflow.log_metric(
        "time/epoch", (time.time() - start) / 60 / training_args.num_train_epochs
    )

    model_path = os.path.join(args.trained_model,"trained_model")
    os.makedirs(model_path, exist_ok=True)

    # saves final model
    trainer.model.save_pretrained(model_path)

    print("trained model saved locally")
    # Registering the model to the workspace
    model = Model.register(
        run.experiment.workspace,
        model_name="hf_mrpc_glue",
        model_path=model_path,
        tags={"type": "huggingface", "task":"glue mrpc"},
        description="Hugingface model finetuned for GLUE mrpc task",
    )


    print("Evaluation...")

    trainer.evaluate()
