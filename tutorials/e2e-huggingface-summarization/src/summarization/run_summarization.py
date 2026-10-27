import logging
import os
import sys
from datasets import load_dataset, load_metric, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    HfArgumentParser,
    IntervalStrategy,
)

import torch
import nltk
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import mlflow
from pynvml import *
import time
from azureml.core import Run
from azureml.core.model import Model


# Input arguments are set with dataclass. Huggingface library stores the default training args in TrainingArguments dataclass
# user args are also defined in dataclasses, we will then load arguments from a tuple of user defined and built-in dataclasses.
@dataclass
class DataArgs:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "name of input HF dataset"}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "config of input HF dataset"}
    )
    text_column: Optional[str] = field(
        default=None, metadata={"help": "the key for text column"}
    )
    summary_column: Optional[str] = field(
        default=None, metadata={"help": "the key for summary column"}
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    preprocessed_datasets: Optional[str] = field(
        default=None, metadata={"help": "path to preprocesed datasets"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "path to train file(json)"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "path to validation file(jsonl)"}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "path to test file(jsonl)"}
    )
    max_input_length: Optional[int] = field(
        default=1024,
        metadata={"help": "max input sequence length after tokenization."},
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={"help": "maxi sequence length for target text after tokenization."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "limit the number of samples for faster run."},
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "limit the number of samples for faster run."},
    )
    padding: Optional[str] = field(
        default="max_length", metadata={"help": "padding setting"}
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.preprocessed_datasets is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name, preprocessed_datasets, or a training/validation file."
            )


@dataclass
class ModelArgs:
    trained_model_path: str
    registered_model_name: Optional[str] = field(
        default=None, metadata={"help": "registered model name"}
    )
    model_name: Optional[str] = field(default=None, metadata={"help": "model name"})
    model_path: Optional[str] = field(
        default=None, metadata={"help": "path to model file"}
    )


run = Run.get_context()
logger = logging.getLogger(__name__)
nltk.download("punkt")


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def main():
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    parser = HfArgumentParser((ModelArgs, DataArgs, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.info(f"Running with arguments: {model_args}, {data_args}, {training_args}")

    # Check if this is the main node
    is_this_main_node = int(os.environ.get("RANK", "0")) == 0
    if is_this_main_node:
        logger.info("This is the main Node")

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        input_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config)
        logger.info(f"dataset {data_args.dataset_name} is loaded")
        preprocessed_flag = False
    elif data_args.preprocessed_datasets is not None:
        input_datasets = load_from_disk(data_args.preprocessed_datasets)
        logger.info(f"preprocessed dataset is loaded")
        preprocessed_flag = True
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        input_datasets = load_dataset("jsonl", data_files=data_files)
        logger.info(f"dataset is loaded from files")
        preprocessed_flag = False

    if model_args.model_path:
        logger.info("using a saved model")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    else:
        logger.info("using a model from model library")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    if (
        data_args.source_prefix is None
        and "t5" in model.config.architectures[0].lower()
    ):
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[data_args.text_column])):
            if (
                examples[data_args.text_column][i] is not None
                and examples[data_args.summary_column][i] is not None
            ):
                inputs.append(examples[data_args.text_column][i])
                targets.append(examples[data_args.summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_input_length,
            padding=data_args.padding,
            truncation=True,
        )
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[data_args.summary_column],
                max_length=data_args.max_target_length,
                padding=data_args.padding,
                truncation=True,
            )

        # replace all tokenizer.pad_token_id in the labels by -100 to ignore padding in the loss.
        if data_args.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if data_args.max_samples > 0:
            max_train_samples = min(len(input_datasets["train"]), data_args.max_samples)
            train_dataset = input_datasets["train"].select(range(max_train_samples))
            logger.info(f"making a {max_train_samples} sample of the data")
        else:
            train_dataset = input_datasets["train"]

        if preprocessed_flag == False:
            # with training_args.main_process_first(desc="train dataset map pre-processing"):
            logger.info(f"tokenizing the train data")
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=train_dataset.column_names,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if data_args.max_samples > 0:
            max_eval_samples = min(
                len(input_datasets["validation"]), data_args.max_samples
            )
            eval_dataset = input_datasets["validation"].select(range(max_eval_samples))
            logger.info(f"making a {max_eval_samples} sample of the data")
        else:
            eval_dataset = input_datasets["validation"]
        if preprocessed_flag == False:
            logger.info(f"tokenizing the evale data")
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Running tokenizer on validation dataset",
            )

    # Data collator
    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    # Metric
    metric = load_metric("rouge")

    if training_args.do_train:
        logging_steps = len(train_dataset) // training_args.per_device_train_batch_size
        training_args.logging_steps = logging_steps
    training_args.output_dir = "outputs"
    training_args.save_strategy = "epoch"
    training_args.evaluation_strategy = IntervalStrategy.EPOCH
    training_args.predict_with_generate = True
    training_args.report_to = ["mlflow"]
    logger.info(f"training args: {training_args}")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    # Training
    if training_args.do_train:
        logger.info("start training")
        start = time.time()
        train_result = trainer.train()

        mlflow.log_metric(
            "time/epoch", (time.time() - start) / 60 / training_args.num_train_epochs
        )
        logger.info(
            "training is done"
        )  # Only print gpu utilization if gpu is available
        if torch.cuda.is_available():
            print_summary(train_result)
        metrics = train_result.metrics

        model_path = os.path.join(model_args.trained_model_path)
        os.makedirs(model_path, exist_ok=True)

        # saves final model
        trainer.save_model(model_path)
        logger.info(f"model is saved at {model_path}")
        print("trained model saved locally")
        # Registering the model to the workspace
        if model_args.registered_model_name is not None and is_this_main_node:
            model = Model.register(
                run.experiment.workspace,
                model_name=model_args.registered_model_name,
                model_path=model_path,
                tags={
                    "type": "huggingface",
                    "task": "summarization",
                    "dataset": f"{data_args.dataset_name}",
                }
                if data_args.dataset_name is not None
                else {"type": "huggingface", "task": "summarization"},
                description=f"Hugingface model finetuned for summarization using {data_args.dataset_name} dataset"
                if data_args.dataset_name is not None
                else "Huggingface model finetuned for summarization",
            )

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=data_args.max_target_length,
            num_beams=data_args.num_beams,
            metric_key_prefix="eval",
        )
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
