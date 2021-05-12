## Introduction

The purpose of these examples is to see how to train huggingface models on Azure ML,
as well as to demonstrate some "real-world" scenarios, e.g.

- Use Huggingface libraries to take pretrained models and finetune them on GLUE benchmarking tasks
- Compare the training times between different Azure VM SKUs
- Perform automatic hyperparameter optimization with Azure ML's HyperDrive library

Note: This is not meant to be an introduction to the Huggingface libraries. In fact, we borrowed
liberally from their example notebooks. You may want to do the same!

- [Huggingface GLUE example notebook](https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb)


## Usage

We provide the following examples:

- `1-aml-finetune-job.py`: Submit single GLUE finetuning script to Azure ML. This script forms
the basis for all other examples.
- `2-aml-comparison-of-sku-job.py`: Experiment comparing training times with different VM SKUs.
- `3-aml-hyperdrive-job.py`: Submit a HyperDrive experiment for automated hyperparameter optimization.

Run these as follows:

```bash
python 1-aml-finetune-job.py
```

Note: Make sure you run this from an environment with azureml-sdk (`pip install azureml-sdk`).

Optionally provide glue task and model checkpoint from the command line:

```bash
# finetune bert-base-cased on rte task
python 1-aml-finetune-job.py --glue_task rte --model_checkpoint bert-base-cased

# compare training times with different VMs
python 2-aml-comparison-of-sku-job.py --glue_task cola --model_checkpoint gpt2

# hyperparameter optimzation with HyperDrive
python 3-aml-hyperdrive-job.py --glue_task mnli --model_checkpoint distilroberta-base
```

Note: Your first run will kick-off an image build: Azure ML is building a docker image with
with the `requirements.txt` installed. This can take 10+ minutes. Future runs will be much faster
as this image is cached and reused.

## Transformers

These examples make use of the Huggingface
[🤗 Transformers library](https://github.com/huggingface/transformers).

Some aspects we make use of here:

### Pretrained Models

We make use of `AutoModelForSequenceClassification.from_pretrained` to load various pretrained
models.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels
)
```

See the full list of pretrained models provided by huggingface
[here](https://huggingface.co/transformers/pretrained_models.html)

### Tokenizers

We make use of the `AutoTokenizer.from_pretrained` method to download a pretrained tokenizer
used to prepare inputs to the model.

```python
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
```

### GLUE Datasets and Metrics

The GLUE benchmarking tasks are available through the Huggingface [🤗 Datasets library](https://github.com/huggingface/datasets). This provides simple APIs that allows us to get the GLUE datasets and metrics:

```python
dataset = load_dataset("glue", task)
metric = load_metric("glue", task)
```

In `glue_datasets.py` we make use of this library to get the raw datasets and metrics, and apply
the pretrained tokenizer to produce the encoded GLUE dataset.

### Azure ML Callback

[Callbacks](https://huggingface.co/transformers/main_classes/callback.html?highlight=callbacks)
are a mechanism that allows customization within the training loop.

Specifically, we make use of the existing [`AzureMLCallback`](https://huggingface.co/transformers/_modules/transformers/integrations.html#AzureMLCallback)
that is used to send logs to Azure ML. This allows us to visualize metrics via the [Studio](https://ml.azure.com)

### HfArgumentParser

In particular the `parse_args_into_dataclasses()` method. The `Trainer` class accepts a dataclass
[`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments)
that packages many arguments used during training e.g. learning rates and batch sizes. By using
`HfArgumentParser` we override fields in `TrainingArgument` from the command-line, while at the
same time specifying our own command-line arguments using the standard `argparse` format.

```python
if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument("--task", default="cola", help="name of GLUE task to compute")
    parser.add_argument("--model_checkpoint", default="distilbert-base-uncased")
    training_args, args = parser.parse_args_into_dataclasses()
```

## Local Training

To test transformers training script locally, create a virtual environment and run:

```bash
pip install -r requirements.txt
cd src
python finetune_glue.py --glue_task "cola" --model_checkpoint "distilbert-base-uncased"
```