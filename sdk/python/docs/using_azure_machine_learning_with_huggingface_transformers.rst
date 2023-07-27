.. _using_azure_machine_learning_with_huggingface_transformers:

Using Azure Machine Learning with Huggingface Transformers
==========================================================

Introduction
------------

The purpose of these examples is to demonstrate how to train Huggingface models on Azure ML, as well as to demonstrate some "real-world" scenarios, such as:

- Using Huggingface libraries to take pretrained models and finetune them on GLUE benchmarking tasks
- Comparing the training times between different Azure VM SKUs
- Performing automatic hyperparameter optimization with Azure ML's HyperDrive library

.. note:: This is not meant to be an introduction to the Huggingface libraries. In fact, we borrowed liberally from their example notebooks. You may want to do the same!

Usage
-----

We provide the following examples:

- `1-aml-finetune-job.py`: Submit single GLUE finetuning script to Azure ML. This script forms the basis for all other examples.
- `2-aml-comparison-of-sku-job.py`: Experiment comparing training times with different VM SKUs.
- `3-aml-hyperdrive-job.py`: Submit a HyperDrive experiment for automated hyperparameter optimization.

Run these as follows:

.. code-block:: bash

   python 1-aml-finetune-job.py

.. note:: Make sure you run this from an environment with azureml-sdk (`pip install azureml-sdk`).

Optionally provide glue task and model checkpoint from the command line:

.. code-block:: bash

   # finetune bert-base-cased on rte task
   python 1-aml-finetune-job.py --glue_task rte --model_checkpoint bert-base-cased

   # compare training times with different VMs
   python 2-aml-comparison-of-sku-job.py --glue_task cola --model_checkpoint gpt2

   # hyperparameter optimzation with HyperDrive
   python 3-aml-hyperdrive-job.py --glue_task mnli --model_checkpoint distilroberta-base

.. note:: Your first run will kick-off an image build: Azure ML is building a docker image with with the `requirements.txt` installed. This can take 10+ minutes. Future runs will be much faster as this image is cached and reused.

Transformers
------------

These examples make use of the Huggingface Transformers library. Some aspects we make use of here include:

- Pretrained Models: We make use of `AutoModelForSequenceClassification.from_pretrained` to load various pretrained models.
- Tokenizers: We make use of the `AutoTokenizer.from_pretrained` method to download a pretrained tokenizer used to prepare inputs to the model.
- GLUE Datasets and Metrics: The GLUE benchmarking tasks are available through the Huggingface Datasets library. This provides simple APIs that allows us to get the GLUE datasets and metrics.

Azure ML Callback
-----------------

Callbacks are a mechanism that allows customization within the training loop. Specifically, we make use of the existing `AzureMLCallback` that is used to send logs to Azure ML. This allows us to visualize metrics via the Azure ML Studio.

HfArgumentParser
----------------

In particular the `parse_args_into_dataclasses()` method. The `Trainer` class accepts a dataclass `TrainingArguments` that packages many arguments used during training e.g. learning rates and batch sizes. By using `HfArgumentParser` we override fields in `TrainingArgument` from the command-line, while at the same time specifying our own command-line arguments using the standard `argparse` format.

Local Training
--------------

To test transformers training script locally, create a virtual environment and run:

.. code-block:: bash

   pip install -r requirements.txt
   cd src
   python finetune_glue.py --glue_task "cola" --model_checkpoint "distilbert-base-uncased"

References
----------

- `Huggingface GLUE example notebook <https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb>`_
- `Huggingface Transformers library <https://github.com/huggingface/transformers>`_
- `Huggingface Datasets library <https://github.com/huggingface/datasets>`_
- `Huggingface Pretrained Models <https://huggingface.co/transformers/pretrained_models.html>`_
- `Huggingface Callbacks <https://huggingface.co/transformers/main_classes/callback.html?highlight=callbacks>`_
- `Huggingface TrainingArguments <https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments>`_