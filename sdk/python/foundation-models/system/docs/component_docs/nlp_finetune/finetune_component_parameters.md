# Finetune Pipeline Component
This component enables finetuning of pretrained models on custom datasets. The component supports Deepspeed for performance enhancement.

The component supports following optimizations:
1. Parameter efficient finetuning with techniques like LoRA.
2. Supports Multi-GPU finetuning using Distributed Data Parallel (DDP) and DeepSpeed.
3. Supports Mixed Precision Training.
4. Supports Multi-Node training.
5. Supports flash attention for speed up finetuning and reducing memory footprint.


At the time of writing, following tasks are supported through finetuning components:
| Task | Notebook |
| --- | --- |
| Text Generation | https://github.com/Azure/azureml-examples/tree/main/sdk/python/foundation-models/system/finetune/Llama-notebooks/text-generation
| Text Classification | https://github.com/Azure/azureml-examples/tree/main/sdk/python/foundation-models/system/finetune/text-classification
| Named Entity Recognition/Token Classification | https://github.com/Azure/azureml-examples/tree/main/sdk/python/foundation-models/system/finetune/token-classification
| Question Answering | https://github.com/Azure/azureml-examples/tree/main/sdk/python/foundation-models/system/finetune/question-answering
| Summarization | https://github.com/Azure/azureml-examples/tree/main/sdk/python/foundation-models/system/finetune/summarization
| Translation | https://github.com/Azure/azureml-examples/tree/main/sdk/python/foundation-models/system/finetune/translation

Respective components can be found in the azureml registry:
- [text_generation_pipeline](https://ml.azure.com/registries/azureml/components/text_generation_pipeline)
- [text_classification_pipeline](https://ml.azure.com/registries/azureml/components/text_classification_pipeline)
- [token_classification_pipeline](https://ml.azure.com/registries/azureml/components/token_classification_pipeline)
- [question_answering_pipeline](https://ml.azure.com/registries/azureml/components/question_answering_pipeline)
- [summarization_pipeline](https://ml.azure.com/registries/azureml/components/summarization_pipeline)
- [translation_pipeline](https://ml.azure.com/registries/azureml/components/translation_pipeline)

# 1. Inputs
## Model related inputs
- _pytorch_model_path_ (custom_model, optional)
    Pytorch model asset path. Special characters like \ and ' are invalid in the parameter value.

- _mlflow_model_path_ (mlflow_model, optional)
    Mlflow model asset path. Special characters like \ and ' are invalid in the parameter value

**Note: one of the above two inputs is required.**

## Dataset related inputs
- _train_file_path_ (uri_file, optional)
    Path to the registered training data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`. Special characters like \ and ' are invalid in the parameter value.

- _validation_file_path_ (uri_file, optional)
    Path to the registered validation data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`. Special characters like \ and ' are invalid in the parameter value.

- _test_file_path_ (uri_file, optional)
    Path to the registered test data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`. Special characters like \ and ' are invalid in the parameter value.

- _train_mltable_path_ (MLTABLE, optional)
    Path to the registered training data asset in `mltable` format. Special characters like \ and ' are invalid in the parameter value.

- _validation_mltable_path_ (MLTABLE, optional)
    Path to the registered validation data asset in `mltable` format. Special characters like \ and ' are invalid in the parameter value.

- _test_mltable_path_ (MLTABLE, optional)
    Path to the registered test data asset in `mltable` format. Special characters like \ and ' are invalid in the parameter value.

## Compute related inputs
- _compute_model_import_ (string, optional, default: "serverless")
    compute to be used for model_import eg. provide 'FT-Cluster' if
      your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value.
      If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used
- _compute_preprocess_ (string, optional, default: "serverless")
    compute to be used for preprocess eg. provide 'FT-Cluster' if your
      compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value.
      If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used
- _compute_finetune_ (string, optional, default: "serverless")
    compute to be used for finetune **NOTE: This has to be GPU compute**
- _compute_model_evaluation_ (string, optional, default: "serverless")
    compute to be used for model_evaluation

### Serverless compute related parameters. Will be used only if the compute used is serverless
- _instance_type_model_import_ (string, optional, default: "Standard_d12_v2")
    Instance type to be used for model_import component in case of serverless compute, eg. standard_d12_v2. 
      The parameter compute_model_import must be set to 'serverless' for instance_type to be used

- _instance_type_preprocess_ (string, optional, default: "Standard_d12_v2")
    Instance type to be used for preprocess component in case of serverless compute, eg. standard_d12_v2. 
      The parameter compute_preprocess must be set to 'serverless' for instance_type to be used

- _instance_type_finetune_ (string, optional, default: "Standard_nc24rs_v3")
    Instance type to be used for finetune component in case of serverless compute, eg. standard_nc6. 
      The parameter compute_finetune must be set to 'serverless' for instance_type to be used

- _instance_type_model_evaluation_ (string, optional, default: "Standard_nc24rs_v3")
    Instance type to be used for model_evaluation component in case of serverless compute, eg. standard_nc6. 
      The parameter compute_model_evaluation must be set to 'serverless' for instance_type to be used

## MultiNode and MultiGPU training parameters
- _number_of_gpu_to_use_finetuning_ (integer, optional, default: 1)
    number of gpus to be used per node for finetuning, should be equal
      to number of gpu per node in the compute SKU used for finetune.

- _num_nodes_finetune_ (integer, optional, default: 1)
    number of nodes to be used for finetuning (used for distributed training).

## Data Preprocessing parameters
- _batch_size_ (integer, optional, default: 1000)
    Number of examples to batch before calling the tokenization function

- _max_seq_length_ (integer, optional, default: -1)
- Default is -1 which means the padding is done up to the model's max length. Else will be padded to `max_seq_length`.

- _pad_to_max_length_ (string, optional, default: "false")
    If set to True, the returned sequences will be padded according to the model's padding side and padding index, up to their `max_seq_length`. If no `max_seq_length` is specified, the padding is done up to the model's max length.

## Finetuning parameters
### LoRA parameters
- _apply_lora_ (string, optional, default: false, allowed_values: [true, false]):
  Whether to enable LoRA for finetuning. If set to true, LoRA will be applied to the model.

- _lora_alpha_ (integer, optional, default: 128)
    lora attention alpha

- _lora_r_ (integer, optional, default: 8)
    lora dimension

- lora_dropout (number, optional, default: 0.0)
    lora dropout value

### Deepspeed parameters
- _apply_deepspeed_ (bool, optional, default: false)
    If true enables deepspeed.

- _deepspeed_ (uri_file, optional, default: true)
  Deepspeed config to be used for finetuning. If no `deepspeed` is provided, the default config in the component will be used else the user passed config will be used.

- _deepspeed_stage_ (string, optional, default: "2")
    Deepspeed stage to be used for finetuning. It could be one of [`2`, `3`]. Value '3' enabled model sharding across GPUs, useful if the model does not fit on a single GPU.

### Training parameters
-  _num_train_epochs_ (int, optional, default: 1)
    Number of epochs to run for finetune.

-  _max_steps_ (int, optional, default: -1)
    If set to a positive number, the total number of training steps to perform. Overrides 'epochs'. In case of using a finite iterable dataset the training may stop before reaching the set number of steps when all data is exhausted.

-  _per_device_train_batch_size_ (integer, optional, default: 1)
    Train batch size

-  _per_device_eval_batch_size_ (integer, optional, default: 1)
    Validation batch size

-  _auto_find_batch_size_ (bool, optional, default: false)
    If set to true, the train batch size will be automatically downscaled recursively till if finds a valid batch size that fits into memory. If the provided 'per_device_train_batch_size' goes into Out Of Memory (OOM) enabling auto_find_batch_size will find the correct batch size by iteratively reducing 'per_device_train_batch_size' by a factor of 2 till the OOM is fixed.

-  _learning_rate_ (number, optional, default: 0.00002)

    Start learning rate used for training. Defaults to linear scheduler.

-  _lr_scheduler_type_ (string, optional, default: linear)

    The learning rate scheduler to use. It could be one of [`linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`].

    If left empty, will be chosen automatically based on the task type and model selected.

-  _warmup_steps_ (integer, optional, default: 0)
    Number of steps used for a linear warmup from 0 to learning_rate.

-  _optim_ (string, optional, default: adamw_hf)

    Optimizer to be used while training. It could be one of [`adamw_hf`, `adamw_torch`, `adafactor`]

    If left empty, will be chosen automatically based on the task type and model selected.

-  _weight_decay_ (number, optional)
    The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer

-  _gradient_accumulation_steps_ (integer, optional, default: 1)

    Number of updates steps to accumulate the gradients for, before performing a backward/update pass

-  _precision_ (string, optional, default: "32")

    Apply mixed precision training. This can reduce memory footprint by performing operations in half-precision. It could one of [`16`, `32`].

-  _seed_ (int, optional, default: 42)

    Random seed that will be set at the beginning of training.

-  _evaluation_strategy_ (string, optional, default: epoch)

    The evaluation strategy to adopt during training. It could be one of [`epoch`, `steps`].

-  _eval_steps_ (int, optional, default: 500)
    Number of update steps between two evals if evaluation_strategy='steps'

-  _logging_strategy_ (string, optional, default: steps)
    The logging strategy to adopt during training. It could be one of [`epoch`, `steps`].

-  _logging_steps_ (integer, optional, default: 10)
    Number of update steps between two logs if logging_strategy='steps'

-  _save_total_limit_ (integer, optional, default: -1)

    If a value is passed, will limit the total number of checkpoints. Deletes the older checkpoints in output_dir. If the value is -1 saves all checkpoints".

-  _apply_early_stopping_ (string, optional, default: "false")
    If set to true, early stopping is enabled. The default value is false.

-  _early_stopping_patience_ (int, optional, default: 1)
    Stop training when the specified metric worsens for early_stopping_patience evaluation calls.

-  _max_grad_norm_ (number, optional, default: 1.0)

    Maximum gradient norm (for gradient clipping).

-  _resume_from_checkpoint_ (string, optional, default: "false")
    If set to true, resumes the training from last saved checkpoint. Along with loading the saved weights, saved optimizer, scheduler and random states will be loaded if exists.

# 2. Outputs
- _pytorch_model_folder_ (uri_folder)
    output folder containing _best_ model as defined by _metric_for_best_model_. Along with the best model, output folder contains checkpoints saved after every evaluation which is defined by the _evaluation_strategy_. Each checkpoint contains the model weight(s), config, tokenizer, optimzer, scheduler and random number states.

- _mlflow_model_folder_ (mlflow_model)
    output folder containing _best_ finetuned model in mlflow format.

- _evaluation_result_ (uri_folder)
    Test Data Evaluation Results