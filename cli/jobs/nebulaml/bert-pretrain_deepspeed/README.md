## Bert Pretraining With Nebula

This example will focus on pretraining a BERT model for Masked Language Modeling (MLM) on the GLUE dataset. Bert is a large model and in this article you can learn on tips and tricks to be able to train with high efficiency for compute and memory without impacting the quality of model.

## Setup:
### Hardware
V100 GPUs (ND40rs) are recommended for this job. This example was originally run using 2 ND40rs nodes with 8 V100 GPUs each.
#### Linear Scaling with Infini band Enabled SKUs
To attain linear scaling for large model, one important step can be to use InfiniBand. InfiniBand enables low-latency, GPU-to-GPU communication across nodes in a cluster. InfiniBand requires specialized hardware to operate. Only some VM SKUs on Azure contain this required hardware. You can view the full list of InfiniBand-enabled machine SKUs [here](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-hpc#rdma-capable-instances). 

### Setup the environment
The environment found at ``src/envrionments`` is an ACPT environment with multiple accelerators to boost the training job and is available out of the box in AzureML. If you would like to add additional packages, edit the appropriate files in ``src/environments`` with your changes, then create the custom environment using the following command:
```
az ml environment create --file ./src/environments/env.yml
```
### Load the dataset
Load the dataset from HuggingFace preprocessed data for GLUE.
```
def load_raw_glue_dataset(task: str) -> Union[DatasetDict, Dataset]:
    dataset = load_dataset("glue", actual_task(task))
    return dataset

def load_encoded_glue_dataset(
    task: str, tokenizer: PreTrainedTokenizerBase
) -> Union[DatasetDict, Dataset]:
    """Load GLUE data, apply tokenizer and split into train/validation."""
    tokenizer_func = construct_tokenizer_function(tokenizer=tokenizer, task=task)
    raw_dataset = load_raw_glue_dataset(task)
    encoded_dataset = raw_dataset.map(tokenizer_func, batched=True)

    validation_key = (
        "validation_mismatched"
        if task == "mnli-mm"
        else "validation_matched"
        if task == "mnli"
        else "validation"
    )
    return encoded_dataset["train"], encoded_dataset[validation_key]
```
  
### Training script overview
The script first loads the dataset using the load_dataset function and then tokenizes the text using the BERT tokenizer provided by the transformers library.
The tokenization is performed using a mapping function that maps the sentences to the tokenized version with or without truncation based on the value of the truncate_longer_samples variable. Then the BertForMaskedLM model is instantiated and trained using the Trainer class.

The TrainingArguments object is used to define the training configuration, including the output directory, the number of training epochs, the batch size, and the evaluation strategy. Finally, the DataCollatorForLanguageModeling is used to create a function to collate the tokenized data and train the MLM model.

### Nebula checkpointing
Nebula checkpoint can be enabled for Pytorch vanilla training as well as Deepspeed.

--save-model parameter makes sure that model parameter status is written to the output directory mounted in the blob. Under the hood, on rerunning the experiment, job checks if checkpoint is available, it resumes from checkpoint and saves the training time significantly.

Add below to the ds_config.json to enable Nebula checkpointing:
```
"nebula": {
        "enabled": true,
        "persistent_storage_path": "/outputs/nebula_checkpoint/",
        "persistent_time_interval": 10,
        "num_of_version_in_retention": 2,
        "enable_nebula_load": true
},
```

After your job runs successfully, you can see below logs in user logs, to check wether checkpoints have been saved successfully by Nebula or not and how much time it takes to save a file in checkpoints.

```
[2023-03-27 03:42:54,860] [INFO] [nebula_checkpoint_engine.py:47:save] [Nebula] Saving pytorch_model.bin under tag checkpoint-20...
[1679888575], size is [219004580], Time difference = 68100[µs]
```

## Running the Job
### Submit with Deepspeed
To try BERT pretraining with DeepSpeed, submit the following command from within this directory:
```
az ml job create --file job.yml
```
To submit with DeepSpeed, we also need to include a ``ds_config.json`` file that specifies the DeepSpeed configuration. The following configuration file was found using DeepSpeed autotuning on bert-large, which is detailed [here](../DeepSpeed-Autotuning/README.md).
```
{
  "train_micro_batch_size_per_gpu": 93,
  "fp16": {
    "enabled": true
  },
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": "outputs/profile.txt"
  },
  "zero_optimization": {
    "stage": 1
  },
  "gradient_accumulation_steps": 1
}
```
> NOTE: Make sure the configurations inside your ``ds_config.json`` file are the same as the equivalent arguments in your ``AML-DeepSpeed-submit.yml`` file. For example, ``train_micro_batch_size_per_gpu`` should have the same value as ``--per_device_train_batch_size``.

Some benefits of using DeepSpeed include:
- DeepSpeed provides features like ZeRO (Zero Redundancy Optimizer) that can help reduce the memory footprint of the model during training. This can be useful when training very large models or working with limited resources.
- With DeepSpeed, you may be able to use larger batch sizes during training, which can help improve the efficiency of the training process. This is because DeepSpeed provides features like ZeRO-Offload, which can reduce the amount of memory needed to store the parameters of the model.

> Running out of memory during training is a common issue in deep learning, To overcome this issue, Deepspeed stage 3 (zero infinity) can be used which offload memory to CPU/NvME disk. It is recommended to start with smaller batch size. A larger batch size requires more memory to process and backpropogate gradients. 

