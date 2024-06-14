[![Python code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
# **Bert Pretraining**

This example will focus on pretraining a BERT model for Masked Language Modeling (MLM) on the combined Wikipedia and bookcorpus dataset. Bert is a large language model and in this article you can learn on tips and tricks to be able to train with high efficiency for compute and memory without impacting the quality of model.

## **Setup**
### **Hardware**
V100 GPUs (STANDARD_ND40RS_V2) are recommended for this job. This example was originally run using 2  STANDARD_ND40RS_V2 nodes with 8 V100 GPUs each.

#### **Linear Scaling with Infiniband Enabled SKUs**
To attain linear scaling for large model, one important step can be to use InfiniBand. InfiniBand enables low-latency, GPU-to-GPU communication across nodes in a cluster. InfiniBand requires specialized hardware to operate. Only some VM SKUs on Azure contain this required hardware. You can view the full list of InfiniBand-enabled machine SKUs [here](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-hpc#rdma-capable-instances). 

### **Environment**
The environment found at ``src/environment`` is an ACPT environment with multiple accelerators to boost the training job. If you would like to add additional packages, edit the appropriate files in ``src/environment`` with your changes, then create the custom environment using the following command:
```
az ml environment create --file ./src/environment/env.yml
```
## **Code**
All of the code described in this document can be found either in one of the submit yml files or in the ``src`` folder of this directory.

### **Job Configuration**
The first step in the training script is to parse the arguments passed in from the command in the ``AML-submit.yml`` file. Most arguments here are specific to the ``TrainerArguments`` class from HuggingFace Transformers (more details [here](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)) and are related to the training loop. However, there are a few here that are important to note when trying out this example:
- ``--tensorboard_log_dir`` This specifies the location where TensorBoard will look for job logs. Make sure this argument matches the value of ``logDir`` under ``my_tensorboard``.
- ``--per_device_train_batch_size`` The batch size for a single step on a GPU. This value should match the value of ``train_micro_batch_size_per_gpu`` in your ``ds_config.json`` file if deepspeed is enabled.
- ``--gradient_accumulation_steps`` Number of training steps to accumulate gradients before using them to compute variables. This value should match the value of ``gradient_accumulation_steps`` in your ``ds_config.json`` file if deepspeed is enabled.
- ``--model_checkpoint`` the model to pretrain. In this case we are pretraining "bert-large-uncased" but this example was also run with DistilBERT and BERT-base. See below for more information.

This example also supports the interactive capabilities from JupyterLab, TensorBoard and VSCode. These are added via the ``services`` section of the yml submit files. For more information on these, see [this](../README.md#interactive-debugging) page. Remove these sections under ``services`` to disable these tools.

#### **DeepSpeed Configuration**
As discussed above, arguments to the command job will need to match arguments in the DeepSpeed configuration file (``ds_config.json``) if DeepSpeed is being used. We use a very simple configuration for this experiment. This config is without the additional profiling + checkpointing tools added to the ``ds_config.json`` located in the ``src`` folder.
```
{
    "train_micro_batch_size_per_gpu": 93,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1
    },
    "gradient_accumulation_steps": 1
}
```
Each setting here is described above, but this configuration also includes ``fp16`` to improve training speed and reduce memory usage. 

This configuration was found by running [DeepSpeed Autotuning](https://www.deepspeed.ai/tutorials/autotuning/) with this training script and BERT large in [this example](../DeepSpeed-Autotuning). DeepSpeed as it relates to this example is described in more detail [here](../README.md#deepspeed).
### **Load the dataset**
Once arguments have been parsed, its time to prepare the dataset. First we prepare a tokenizer to tokenize the data:
```
tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
```
This tokenizer will be created based on the value of ``args.model_checkpoint``. In this case, ``BertTokenizer`` from HuggingFace Transformers will be used. To ready the data for training, the tokenizer will split the data into words or subwords (tokens) and encode them by converting them to integers.

Next we load the dataset from HuggingFace preprocessed data for Wikipedia + Bookcorpus.
```
def load_encoded_wiki_dataset(
    tokenizer: PreTrainedTokenizerBase
) -> Union[DatasetDict, Dataset]:

    """Load wiki + corpus data, apply tokenizer and split into train/validation."""
    # Construct tokenizer function
    tokenizer_func = construct_tokenizer_function(tokenizer=tokenizer)
    # load raw data
    raw_wiki_dataset = load_raw_wiki_dataset()
    raw_corpus_dataset = load_raw_corpus_dataset()
    assert raw_corpus_dataset.features.type == raw_wiki_dataset.features.type
    # Combine datasets
    full_raw_dataset = interleave_datasets([raw_corpus_dataset, raw_wiki_dataset])
    # tokenize dataset
    encoded_dataset = full_raw_dataset.map(tokenizer_func, batched=True, remove_columns=["text"])
    # Prepare function for grouping text into batches
    group_texts=construct_group_texts(tokenizer=tokenizer)
    # Batch the data
    encoded_dataset = encoded_dataset.map(group_texts, batched=True)
    return encoded_dataset
```
This is done from within the [``wiki_datasets.py``](./src/glue_datasets.py) file.

### **Train the Model**
Next we train the model, but first we prepare the model configuration:
```
model_config = AutoConfig.from_pretrained(
    args.model_checkpoint,
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = AutoModelForMaskedLM.from_config(model_config)
encoded_dataset_train = load_encoded_wiki_dataset(
    tokenizer=tokenizer
)
```
Since this is a pretraining job and we don't want a pretrained model already, the model will be created from a config instead of using a ``from_pretrained()`` method like the tokenizer. In this case we create a ``BertForMaskedLM`` model.

Finally, we create the Trainer and train the model. Instead of a visible training loop, the ``Trainer.train()`` method will internally execute the loop.
```

trainer = ORTTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset_train,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[AzureMLCallback, ProfilerCallback]
)

```
The ``ProfilerCallback`` in the above code is used to integrate the experiment with [Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). For more information on this code, see [this page](../README.md#pytorch-profiler).

## **Run the Job**
### **Submit with vanilla Pytorch**
To try BERT pretraining with vanilla Pytorch, submit the following command from within this directory:
```
az ml job create --file AML-submit.yml
```
### **Submit with Deepspeed + ORT**
To try BERT pretraining with DeepSpeed and ORT, submit the following command from within this directory:
```
az ml job create --file AML-DeepSpeed-submit.yml
```

## Results

Some results achieved using this example:
| Batch Size | Gradient Accumulation Steps | Time | Loss | Throughput |
|------------|------------------|------|------|-----------|
|64 |32 |23h 1min |2.46 |3712 |
|16 |32 |25h 46min |2.49 |2064 |
|64 |128 |21h 19min |2.34 |3713 |