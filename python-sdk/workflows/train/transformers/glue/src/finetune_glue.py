import argparse
import numpy as np
import mlflow
import time
from typing import Any, List, Union, Dict, Callable

# from dataclasses import dataclass, field
from datasets import Metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
)
from glue_datasets import (
    load_encoded_glue_dataset,
    num_labels_from_task,
    load_metric_from_task,
)

# Azure ML imports - could replace this with e.g. wandb or mlflow
from transformers.integrations import MLflowCallback


def construct_compute_metrics_function(task: str) -> Callable[[EvalPrediction], Dict]:
    metric = load_metric_from_task(task)

    if task != "stsb":

        def compute_metrics_function(eval_pred: EvalPrediction) -> Dict:
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)

    else:

        def compute_metrics_function(eval_pred: EvalPrediction) -> Dict:
            predictions, labels = eval_pred
            predictions = predictions[:, 0]
            return metric.compute(predictions=predictions, references=labels)

    return compute_metrics_function


if __name__ == "__main__":

    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument("--task", default="cola", help="name of GLUE task to compute")
    parser.add_argument("--model_checkpoint", default="distilbert-base-uncased")
    training_args, args = parser.parse_args_into_dataclasses()

    task: str = args.task.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)

    encoded_dataset_train, encoded_dataset_eval = load_encoded_glue_dataset(
        task=task, tokenizer=tokenizer
    )

    num_labels = num_labels_from_task(task)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, num_labels=num_labels
    )

    compute_metrics = construct_compute_metrics_function(args.task)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.pop_callback(MLflowCallback)

    print("Training...")

    start = time.time()
    trainer.train()
    mlflow.log_metric(
        "time/epoch", (time.time() - start) / 60 / training_args.num_train_epochs
    )

    print("Evaluation...")

    trainer.evaluate()
