# -------------------------------------------------------------------------
# Portions Copyright (c) Microsoft Corporation.  All rights reserved.
# --------------------------------------------------------------------------
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import mlflow
import time
from typing import Dict, Callable
import json
import os

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

# Pytorch Profiler
import torch.profiler.profiler as profiler
from transformers import TrainerCallback

# Onnx Runtime for training
from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments


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
    parser = HfArgumentParser(ORTTrainingArguments)
    parser.add_argument("--task", default="cola", help="name of GLUE task to compute")
    parser.add_argument("--model_checkpoint", default="bert-large-uncased")
    parser.add_argument("--tensorboard_log_dir", default="/outputs/runs/")

    training_args, args = parser.parse_args_into_dataclasses()

    transformers.logging.set_verbosity_debug()

    task: str = args.task.lower()

    num_labels = num_labels_from_task(task)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    context_length = 512

    model_config = AutoConfig.from_pretrained(
        args.model_checkpoint,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AutoModelForSequenceClassification.from_config(model_config)

    encoded_dataset_train, encoded_dataset_eval = load_encoded_glue_dataset(
        task=task, tokenizer=tokenizer
    )

    compute_metrics = construct_compute_metrics_function(args.task)

    # Create path for logging to tensorboard
    my_logs = os.environ["PWD"] + args.tensorboard_log_dir

    # Custom HuggingFace trainer callback used for starting/stopping the pytorch profiler
    class ProfilerCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.prof = profiler.profile(
                schedule=profiler.schedule(wait=2, warmup=1, active=3, repeat=2),
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=profiler.tensorboard_trace_handler(my_logs),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )
            self.prof.start()

        def on_train_end(self, args, state, control, model=None, **kwargs):
            self.prof.stop()

        def on_step_begin(self, args, state, control, model=None, **kwargs):
            self.prof.step()

    # Initialize huggingface trainer. This trainer will internally execute the training loop
    trainer = ORTTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_eval,
        # data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[ProfilerCallback],
        feature="sequence-classification",
    )

    trainer.pop_callback(MLflowCallback)

    start = time.time()

    # pretrian the model!
    result = trainer.train()

    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print("Training...")

    mlflow.log_metric(
        "time/epoch", (time.time() - start) / 60 / training_args.num_train_epochs
    )

    print("Evaluation...")

    trainer.evaluate()
