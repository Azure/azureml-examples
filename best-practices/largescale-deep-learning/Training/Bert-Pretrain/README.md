# **Bert Pretraining**

This example will focus on pretraining a BERT model for Masked Language Modeling (MLM) on the GLUE dataset. Bert is a large model and in this article you can learn on tips and tricks to be able to train with high efficiency for compute and memory without impacting the quality of model.

## **Setup**
### **Hardware**
V100 GPUs (STANDARD_ND40RS_V2) are recommended for this job. This example was originally run using 2  STANDARD_ND40RS_V2 nodes with 8 V100 GPUs each.

#### **Linear Scaling with Infiniband Enabled SKUs**
To attain linear scaling for large model, one important step can be to use InfiniBand. InfiniBand enables low-latency, GPU-to-GPU communication across nodes in a cluster. InfiniBand requires specialized hardware to operate. Only some VM SKUs on Azure contain this required hardware. You can view the full list of InfiniBand-enabled machine SKUs [here](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-hpc#rdma-capable-instances). 

### **Environment**
The environment found at ``src/envrionment`` is an ACPT environment with multiple accelerators to boost the training job. If you would like to add additional packages, edit the appropriate files in ``src/environment`` with your changes, then create the custom environment using the following command:
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

Next we load the dataset from HuggingFace preprocessed data for GLUE.
```
encoded_dataset_train, encoded_dataset_eval = load_encoded_glue_dataset(
    task=task, tokenizer=tokenizer
)
```
This is done from within the [``glue_datasets.py``](../src/glue_datasets.py) file.
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
Before training the model, we also load a metric function specific to the GLUE dataset task we perform. The default task is [CoLA](https://nyu-mll.github.io/CoLA/).
```
compute_metrics = construct_compute_metrics_function(args.task)
```
### **Train the Model**
Next we train the model, but first we prepare the model configuration:
```
model_config = AutoConfig.from_pretrained(
    args.model_checkpoint,
    vocab_size=len(tokenizer),
    n_ctx=context_length, # context_length = 512; Value specific to BERT Large
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = AutoModelForSequenceClassification.from_config(model_config)
```
Since this is a pretraining job and we don't want a pretrained model already, the model will be created from a config instead of using a ``from_pretrained()`` method like the tokenizer. In this case we create a ``BertForMaskedLM`` model.

Finally, we create the Trainer and train the model. Instead of a visible training loop, the ``Trainer.train()`` method will internally execute the loop.
```
trainer = Trainer(
    model,
    training_args,
    train_dataset=encoded_dataset_train,
    eval_dataset=encoded_dataset_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[ProfilerCallback]
)

trainer.pop_callback(MLflowCallback)

result = trainer.train()
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

## **Results**
### **Recorded Metrics**

- **Training time**: This refers to the total time taken to train the model from start to finish. Using DeepSpeed may complete the training process in less time.

- **Training throughput**: This metric refers to the number of training examples that can be processed per second during training. With DeepSpeed, you may be able to achieve higher training throughput, which can help speed up the training process.

These metrics were calculated while using two ND40rs compute nodes with 8 V100 gpus each.
|      Metrics      |   Vanilla Pytorch     | DeepSpeed | Improvement |
| ----------------- | --------------- | ------------------ |----------|
| Training Time     |      351.75 s   |   253.79 s     |    27.8%     |
| samples/second    |      2431.02    |   3369.37      | 27.8%        |

## **Additional Experiments**

In addition to BERT-large, this repository also contains code for pretraining for both DistilBERT and BERT-base models. These experiments can be run with the same setup as BERT-large.

### **BERT-base**

To pretrain the BERT-base model using vanilla pytorch, edit the ``AML-submit.yml`` file and replace the existing command with the following:
```
python pretrain_glue.py --num_train_epochs 100 --output_dir outputs --disable_tqdm 1 --local_rank $RANK --evaluation_strategy "epoch" --logging_strategy "epoch" --per_device_train_batch_size 128 --gradient_accumulation_steps 1 --per_device_eval_batch_size 128 --learning_rate 3e-05 --adam_beta1 0.8 --adam_beta2 0.999 --weight_decay 3e-07 --warmup_steps 500 --fp16 --logging_steps 1000 --model_checkpoint "bert-base-uncased"
```
To pretrain using the optimal DeepSpeed configuration file found by DeepSpeed Autotuning, edit the ``AML-DeepSpeed-submit.yml`` file and replace the existing command with the following:
```
python pretrain_glue.py --deepspeed ds_config_bertbase.json --num_train_epochs 100 --output_dir outputs --disable_tqdm 1 --local_rank $RANK --evaluation_strategy "epoch" --logging_strategy "epoch" --per_device_train_batch_size 532 --gradient_accumulation_steps 1 --per_device_eval_batch_size 532 --learning_rate 3e-05 --adam_beta1 0.8 --adam_beta2 0.999 --weight_decay 3e-07 --warmup_steps 500 --fp16 --logging_steps 1000 --model_checkpoint "bert-base-uncased"
```

### **DistilBERT**

To pretrain the BERT-base model using vanilla pytorch, edit the ``AML-submit.yml`` file and replace the existing command with the following:
```
python pretrain_glue.py --num_train_epochs 100 --output_dir outputs --disable_tqdm 1 --local_rank $RANK --evaluation_strategy "epoch" --logging_strategy "epoch" --per_device_train_batch_size 256 --gradient_accumulation_steps 1 --per_device_eval_batch_size 256 --learning_rate 3e-05 --adam_beta1 0.8 --adam_beta2 0.999 --weight_decay 3e-07 --warmup_steps 500 --fp16 --logging_steps 1000 --model_checkpoint "distilbert-base-uncased"
```
To pretrain using the optimal DeepSpeed configuration file found by DeepSpeed Autotuning, edit the ``AML-DeepSpeed-submit.yml`` file and replace the existing command with the following:
```
python pretrain_glue.py --deepspeed ds_config_distilbert.json --num_train_epochs 100 --output_dir outputs --disable_tqdm 1 --local_rank $RANK --evaluation_strategy "epoch" --logging_strategy "epoch" --per_device_train_batch_size 512 --gradient_accumulation_steps 2 --per_device_eval_batch_size 512 --learning_rate 3e-05 --adam_beta1 0.8 --adam_beta2 0.999 --weight_decay 3e-07 --warmup_steps 500 --fp16 --logging_steps 1000 --model_checkpoint "distilbert-base-uncased"
```

### **Result Comparison**
#### **DistilBERT**
| Optimizations  | Model size  | GPU  | MBS  | Samples/Second  | Improvement |
|----------------|-------------|------|------|-----------------|-------------|
| Vanilla Pytorch| 66M         | 16   | 256  | 12373.53        |     --      |
| DeepSpeed + Autotuning| 66M  | 16   | 512  | 14419.00        | 14%         |

#### **BERT-base**
| Optimizations  | Model size  | GPU  | MBS  | Samples/Second  | Improvement |
|----------------|-------------|------|------|-----------------|-------------|
| Vanilla Pytorch| 110M        | 16   | 128  | 7837.66         |     --      |
| DeepSpeed + Autotuning| 110M | 16   | 532  | 9916.19         | 21%         |

#### **BERT-large**
| Optimizations  | Model size        | GPU  | MBS  | Samples/Second  | Improvement |
|----------------|-------------------|------|------|-----------------|-------------|
| Vanilla Pytorch| 330M              | 16   | 64   | 2431.02         |     --      |
| DeepSpeed + Autotuning| 330M | 16   | 93   | 3369.37         | 28%         |
