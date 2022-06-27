import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, load_metric
import os
import argparse
import logging
from azureml.core import Run
import mlflow


logger = logging.getLogger(__name__)
nltk.download("punkt")
rouge_score = load_metric("rouge")

def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


def evaluate_baseline(datasets, metric, text_column, summary_column):
    summaries = [three_sentence_summary(text) for text in datasets[text_column]]
    return metric.compute(predictions=summaries, references=datasets[summary_column])

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
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="dataset_name")
    parser.add_argument("--dataset_config", type=str, help="dataset_config")
    parser.add_argument("--text_column", type=str, help="text_column")
    parser.add_argument("--summary_column", type=str, help="summary_column")
    parser.add_argument("--max_samples", type=int, required=False, default=-1, help="max samples to be used for evaluation")
    args = parser.parse_args()

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config)
    if args.max_samples > 0:
        max_eval_samples = min(
            len(raw_datasets["validation"]), args.max_samples
        )
        eval_dataset = raw_datasets["validation"].select(range(max_eval_samples))
        logger.info(f"making a {max_eval_samples} sample of the data")
    else:
        eval_dataset = raw_datasets["validation"]

    score = evaluate_baseline(eval_dataset, rouge_score, args.text_column, args.summary_column)
    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
    logger.info(rouge_dict)
    for name in rouge_names:
        mlflow.log_metric(
        name, rouge_dict[name]
    )

if __name__ == "__main__":
    main()