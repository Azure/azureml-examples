import numpy as np
import mlflow
import time
from typing import Dict, Callable
import json

# from dataclasses import dataclass, field
import transformers
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

# pretraining
from transformers import AutoConfig
from transformers import DataCollatorForLanguageModeling

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
    parser.add_argument("--model_checkpoint", default="bert-large-uncased")
    training_args, args = parser.parse_args_into_dataclasses()

    transformers.logging.set_verbosity_debug()

    # with open('ds_config.json') as f:
    #     my_config = json.load(f)

    task: str = args.task.lower()

    num_labels = num_labels_from_task(task)

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     args.model_checkpoint, num_labels=num_labels
    # )

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    context_length = 512
    # tokenizer.pad_token = tokenizer.eos_token
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model_config = AutoConfig.from_pretrained(
        args.model_checkpoint,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AutoModelForSequenceClassification.from_config(model_config)
    # model.config.pad_token_id = model.config.eos_token_id
    # if tokenizer.pad_token is None:
    #     print("adding pad_token!")
    #     # note: adding new pad token will change the vocab size
    #     # to keep it simple just reuse an existing special token
    #     # https://github.com/huggingface/transformers/issues/6263
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = model.config.eos_token_id

    encoded_dataset_train, encoded_dataset_eval = load_encoded_glue_dataset(
        task=task, tokenizer=tokenizer
    )

    compute_metrics = construct_compute_metrics_function(args.task)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_eval,
        # data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.pop_callback(MLflowCallback)

    start = time.time()
    result = trainer.train()

    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print("Training...")

    mlflow.log_metric(
        "time/epoch", (time.time() - start) / 60 / training_args.num_train_epochs
    )

    print("Evaluation...")

    trainer.evaluate()
