import os
import argparse
import pandas as pd
import logging
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import mlflow


logger = logging.getLogger(__name__)


def main():
    """Main function of the script."""
    # initialize root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, help="name of dataset or path to input dataset_name"
    )
    parser.add_argument(
        "--dataset_config", type=str, help="config for huggingface dataset"
    )
    parser.add_argument("--text_column", type=str, help="name of text_column")
    parser.add_argument("--summary_column", type=str, help="name of summary_column")
    parser.add_argument(
        "--max_input_length", type=int, default=512, help="max_input_length"
    )
    parser.add_argument(
        "--max_target_length", type=int, default=40, help="max_target_length"
    )
    parser.add_argument(
        "--padding", type=str, default="max_length", help="padding type"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="name of the checkpointed model in HF model library",
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1, help="sample size from input dataset"
    )
    parser.add_argument("--encodings", type=str, help="path to tokenized dataset")
    parser.add_argument(
        "--source_prefix",
        type=str,
        help="A prefix to add before every source text (useful for T5 models).",
    )

    args = parser.parse_args()
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length

    # Start Logging
    mlflow.start_run()

    logger.info(f"Running with arguments: {args}")

    # Load dataset
    raw_dataset = load_dataset(args.dataset_name, args.dataset_config)

    logger.info(f"raw dataset length: {raw_dataset.num_rows}")

    mlflow.log_metric("train_samples", raw_dataset["train"].shape[0])

    if args.max_samples > 0:
        sample_sizes = {
            k: min(len(raw_dataset[k]), args.max_samples) for k in raw_dataset.keys()
        }
        raw_dataset = DatasetDict(
            {
                k: raw_dataset[k].select([i for i in range(sample_sizes[k])])
                for k in raw_dataset.keys()
            }
        )
        logger.info("sampled raw dataset:")
        logger.info(raw_dataset.num_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    logger.info(f"tokenizer: {tokenizer}")

    prefix = args.source_prefix if args.source_prefix is not None else ""
    if args.source_prefix is None and "t5" in args.model_checkpoint.lower():
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[args.text_column])):
            if (
                examples[args.text_column][i] is not None
                and examples[args.summary_column][i] is not None
            ):
                inputs.append(examples[args.text_column][i])
                targets.append(examples[args.summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_input_length,
            padding=args.padding,
            truncation=True,
        )
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[args.summary_column],
                max_length=args.max_target_length,
                padding=args.padding,
                truncation=True,
            )

        # replace all tokenizer.pad_token_id in the labels by -100 to ignore padding in the loss.
        if args.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    preprocessed_datasets = raw_dataset.map(preprocess_function, batched=True)
    logger.info(f"preprocessed_datasets: {preprocessed_datasets}")

    output_path = os.path.join(args.encodings)
    os.makedirs(output_path, exist_ok=True)

    preprocessed_datasets.save_to_disk(output_path)

    logger.info(f"tokenized data is saved to {output_path}")

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
