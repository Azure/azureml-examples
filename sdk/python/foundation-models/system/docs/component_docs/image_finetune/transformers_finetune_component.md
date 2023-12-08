# Transformers Model Finetune Component
This component enables finetuning of pretrained models on custom or pre-available datasets. The component supports Deepspeed and ONNXRuntime configurations for performance enhancement. The components can be seen in your workspace component page as below:
- [transformers_image_classification_model_import](https://ml.azure.com/registries/azureml/components/transformers_image_classification_model_import)
- [transformers_image_classification_finetune](https://ml.azure.com/registries/azureml/components/transformers_image_classification_finetune)
- [transformers_image_classification_pipeline](https://ml.azure.com/registries/azureml/components/transformers_image_classification_pipeline)

# 1. Inputs
1. _model_path_ (URI_FOLDER, required)

    Path to the output directory of [model import component](transformers_model_import_component.md/#2-outputs).

2. _training_data_ (MLTABLE, required)

    Path to the mltable folder of training dataset.

3. _validation_data_ (MLTABLE, optional)

    Path to the mltable folder of validation dataset.

4. _image_height_ (int, optional)

    Final Image height after augmentation that is input to the network. Default value is -1 which means it would be overwritten by default image height in Hugging Face feature extractor. If either image_width or image_height is set to -1, default value would be used for both width and height.

5. _image_width_ (int, optional)

    Final Image width after augmentation that is input to the network. Default value is -1 which means it would be overwritten by default image width in Hugging Face feature extractor. If either image_width or image_height is set to -1, default value would be used for both width and height.

6. _task_name_ (string, required)

    Which task the model is solving.
    It could be one of [`image-classification`, `image-classification-multilabel`].

7. _metric_for_best_model_ (string, optional)

    Specify the metric to use to compare two different models. If left empty, will be chosen automatically based on the task type selected. It could be one of [`loss`, `f1_score_macro`, `accuracy`, `precision_score_macro`, `recall_score_macro`, `iou`, `iou_macro`, `iou_micro`, `iou_weighted`].

    If selecting by yourself, use iou_* metrics in case of multi-label classification task.
    Generally, `accuracy` is chosen for multi-class classification task, and `iou` is chosen for multi-label classification task.

8. _apply_augmentations_ (bool, optional)

    If set to true, will enable data augmentations for training.
    The default value is true.

9. _number_of_workers_ (int, optional)

    Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process. The default value is 8.

10. _apply_deepspeed_ (bool, optional)

    If true enables deepspeed. If no `deepspeed_config` is provided, the default config in the component will be used else the user passed config will be used. The default value is false.

    Please note that to enable deepspeed, `apply_deepspeed` must be set to true.

11. _deepspeed_config_ (URI_FILE, optional)

    Path to the deepspeed config file.

12. _apply_ort_ (bool, optional)

    If true apply ORT optimization. The default value is false.

13. _number_of_epochs_ (int, optional)

    Number of epochs to run for finetune.

    If left empty, will be chosen automatically based on the task type and model selected.

14. _max_steps_ (int, optional)

    If set to a positive number, it's the total number of training steps to perform. It overrides `number_of_epochs`. In case of using a finite iterable dataset the training may stop before reaching the set number of steps when all data is exhausted.

    If left empty, will be chosen automatically based on the task type and model selected.

15. _training_batch_size_ (int, optional)

    Batch size used for training.

    If left empty, will be chosen automatically based on the task type and model selected.

16. _validation_batch_size_ (int, optional)

    Batch size used for validation.

    If left empty, will be chosen automatically based on the task type and model selected.

17. _auto_find_batch_size_ (bool, optional)

    If set to true, the train batch size will be automatically downscaled recursively till if finds a valid batch size that fits into memory. If the provided 'per_device_train_batch_size' goes into Out Of Memory (OOM) enabling auto_find_batch_size will find the correct batch size by iteratively reducing 'per_device_train_batch_size' by a factor of 2 till the OOM is fixed. The default value is false.

18. _learning_rate_ (float, optional)

    Start learning rate used for training.

    If left empty, will be chosen automatically based on the task type and model selected.

19. _learning_rate_scheduler_ (string, optional)

    The learning rate scheduler to use. It could be one of [`warmup_linear`, `warmup_cosine`, `warmup_cosine_with_restarts`, `warmup_polynomial`, `constant`, `warmup_constant`].

    If left empty, will be chosen automatically based on the task type and model selected.

20. _warmup_steps_ (int, optional)

    The number of steps for the learning rate scheduler warmup phase.

    If left empty, will be chosen automatically based on the task type and model selected.

21. _optimizer_ (string, optional)

    Optimizer to be used while training. 'adamw_ort_fused' optimizer is only supported for ORT training. It could be one of [`adamw_hf`, `adamw`, `sgd`, `adafactor`, `adagrad`, `adamw_ort_fused`]

    If left empty, will be chosen automatically based on the task type and model selected.

22. _weight_decay_ (float, optional)

    The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.

    If left empty, will be chosen automatically based on the task type and model selected.

23. _extra_optim_args_: (string, optional)

    Optional additional arguments that are supplied to SGD Optimizer. The arguments should be semi-colon separated key value pairs and should be enclosed in double quotes. For example, "momentum=0.5; nesterov=True" for sgd. Please make sure to use a valid parameter names for the chosen optimizer. For exact parameter names, please refer to https://pytorch.org/docs/1.13/generated/torch.optim.SGD.html#torch.optim.SGD for SGD. Parameters supplied in extra_optim_args will take precedence over the parameter supplied via other arguments such as weight_decay. If weight_decay is provided via "weight_decay" parameter and via extra_optim_args both, values specified in extra_optim_args will be used.

24. _gradient_accumulation_step_ (int, optional)

    Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

    If left empty, will be chosen automatically based on the task type and model selected.

25. _precision_ (int, optional)

    Apply mixed precision training. This can reduce memory footprint by performing operations in half-precision. It could one of [`16`, `32`].

    The default value is "32".

26. _label_smoothing_factor_ (float, optional)

    The label smoothing factor to use in range `[0.0, 1,0)`. Zero means no label smoothing, otherwise the underlying onehot-encoded labels are changed from 0s and 1s to label_smoothing_factor/num_labels and 1 - label_smoothing_factor + label_smoothing_factor/num_labels respectively. Not applicable to multi-label classification.

    If left empty, will be chosen automatically based on the task type and model selected.

27. _random_seed_ (int, optional)

    Random seed that will be set at the beginning of training. The default value is 42.

28. _evaluation_strategy_ (string, optional)

    The evaluation strategy to adopt during training. If set to "steps", either the `evaluation_steps_interval` or `evaluation_steps` needs to be specified, which helps to determine the step at which the model evaluation needs to be computed else evaluation happens at end of each epoch. The default value is "epoch".
    It could be one of [`epoch`, `steps`].

29. _evaluation_steps_ (int, optional)

    Number of update steps between two evals if evaluation_strategy='steps'. The default value is 500.

30. _logging_strategy_ (string, optional)

    The logging strategy to adopt during training. If set to "steps", the `logging_steps` will decide the frequency of logging else logging happens at the end of epoch. The default value is "epoch". It could be one of [`epoch`, `steps`].

31. _logging_steps_ (int, optional)

    Number of update steps between two logs if logging_strategy='steps'. The default value is 500.

32. _save_strategy_ (string, optional)

    The checkpoint save strategy to adopt during training.
    The default value is "epoch".
    It could be one of [`epoch`, `steps`].

33. _save_steps_ (int, optional)

    Number of updates steps before two checkpoint saves if save_strategy="steps".
    The default value is 500.

34. _save_total_limit_ (int, optional)

    If a value is passed, will limit the total number of checkpoints. Deletes the older checkpoints in output_dir. If the value is -1 saves all checkpoints". The default value is -1.

35. _early_stopping_ (bool, optional)

    If set to true, early stopping is enabled. The default value is false.

36. _early_stopping_patience_ (int, optional)

    Stop training when the specified metric doesn't improve for early_stopping_patience evaluation calls. The default value is 1.

37. _max_grad_norm_ (float, optional)

    Maximum gradient norm (for gradient clipping).

    If left empty, will be chosen automatically based on the task type and model selected.

38. _resume_from_checkpoint_ (bool, optional)

    If set to true, resumes the training from last saved checkpoint. Along with loading the saved weights, saved optimizer, scheduler and random states will be loaded if exists. The default value is false.

39. _save_as_mlflow_model_ (bool, optional)

    Save as mlflow model with pyfunc as flavour. The default value is true.

# 2. Outputs
1. _output_dir_pytorch_ (custom_model, required)

    The folder containing finetuned model output with checkpoints, model config, tokenizer, optimizer and scheduler states and random number states in case of distributed training.

2. _output_dir_mlflow_ (URI_FOLDER, optional)

    Output directory to save the finetuned model as mlflow model.

# 4. Run Settings

This setting helps to choose the compute for running the component code. **For the purpose of finetune, gpu compute should be used**. We recommend using Standard_NC6s or Standard_NC6s_v3 compute.

> Select *Use other compute target*

- Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and the corresponding instance / cluster created in your workspace.
- If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute. See the figure below
![other compute target](../../images/other_compute_target_for_image_components.png)

## 4.1. Settings for Distributed Training

> When creating the compute, set the `Maximum number of nodes` to the desired value for multi-node training as shown in the figure below

![set maximum nodes](../../images/maximum_num_nodes.png)

> In case of distributed training, a.k.a multi-node training, the mode must be set to `Mount` (not `Upload`) as shown in the figure below

![Output settings finetune](../../images/image_classification_output_settings.png)

> Set the number of processes under Distribution subsection to use all the gpus in a node

To use all the gpus within a node, set the `Process count per instance` to number of gpus in that node as shown below

![process count per instance](../../images/process_count_per_instance.png)

> Set the number of nodes under the Resources subsection

In case of distributed training, you can configure `instance count` under this subsection to increase the number of nodes as shown below

![instance count](../../images/instance_count.png)