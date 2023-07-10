# Transformers Model Finetune Component
This component enables finetuning of pretrained models on custom or pre-available datasets. The component supports Deepspeed and ONNXRuntime configurations for performance enhancement. 
The components can be seen here ![as shown in the figure](../../images/image_classification_transformers_finetune_components.jpg)

# 1. Inputs
1. _model_path_ (URI_FOLDER, required)

    Path to the output directory of [model import component](transformers_model_import_component.md/#2-outputs).

2. _training_data_ (MLTABLE, required)

    Path to the mltable folder of training dataset.

3. _validation_data_ (MLTABLE, optional)

    Path to the mltable folder of validation dataset.

4. _auto_hyperparameter_selection_ (bool, optional)

    If set to true, will automatically choose the best hyperparameters for the given model and will ignore the hyperparameters provided by the user. The default value is false.

5. _image_height_ (int, optional)

    Final Image height after augmentation that is input to the network. The default value is 224.

6. _image_width_ (int, optional)

    Final Image width after augmentation that is input to the network. The default value is 224.

7. _task_name_ (string, required)

    Which task the model is solving.
    It could be one of [`image-classification`, `image-classification-multilabel`].

8. _apply_augmentations_ (bool, optional)

    If set to true, will enable data augmentations for training and validation.
    The default value is false.

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

    Number of epochs to run for finetune. The default value is 15.

14. _max_steps_ (int, optional)

    If set to a positive number, it's the total number of training steps to perform. It overrides `epochs`. In case of using a finite iterable dataset the training may stop before reaching the set number of steps when all data is exhausted. The default value is -1.

15. _training_batch_size_ (int, optional)

    Batch size used for training. The default value is 4.

16. _validation_batch_size_ (int, optional)

    Batch size used for validation. The default value is 4.

17. _auto_find_batch_size_ (bool, optional)

    If set to true, the train batch size will be automatically downscaled recursively till if finds a valid batch size that fits into memory. The default value is false.

18. _learning_rate_ (float, optional)

    Start learning rate used for training. The default value is 5e-5.

19. _learning_rate_scheduler_ (string, optional)

    The learning rate scheduler to use. The default value is warmup_linear.
    It could be one of [`warmup_linear`, `warmup_cosine`, `warmup_cosine_with_restarts`, `warmup_polynomial`,
    `constant`, `warmup_constant`]

20. _warmup_steps_ (int, optional)
    
    The number of steps for the learning rate scheduler warmup phase. The default value is 0.

21. _optimizer_ (string, optional)

    Optimizer to be used while training. The default value is adamw_hf.
    It could be one of [`adamw_hf`, `adamw`, `sgd`, `adafactor`, `adagrad`]

22. _weight_decay_ (float, optional)

    The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer. The default value is 0.

23. _gradient_accumulation_step_ (int, optional)

    Number of updates steps to accumulate the gradients for, before performing a backward/update pass. The default value is 1.

24. _precision_ (int, optional)

    Apply mixed precision training. This can reduce memory footprint by performing operations in half-precision. The default value is 32.
    It could one of [`16`, `32`]

25. _metric_for_best_model_ (string, optional)

    Specify the metric to use to compare two different models. The default value is accuracy.

26. _label_smoothing_factor_ (float, optional)

    The label smoothing factor to use in range `[0.0, 1,0)`. Zero means no label smoothing, otherwise the underlying onehot-encoded labels are changed from 0s and 1s to label_smoothing_factor/num_labels and 1 - label_smoothing_factor + label_smoothing_factor/num_labels respectively. Not applicable to multi-label classification.

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

    If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir. If the value is -1 saves all checkpoints". The default value is -1.

35. _early_stopping_ (bool, optional)

    If set to true, early stopping is enabled. The default value is false.

36. _early_stopping_patience_ (int, optional)

    Stop training when the specified metric worsens for early_stopping_patience evaluation calls. The default value is 1.

37. _max_grad_norm_ (float, optional)

    Maximum gradient norm (for gradient clipping). The deafult value is 1.0.

38. _resume_from_checkpoint_ (bool, optional)

    If set to true, resumes the training from last saved checkpoint. Along with loading the saved weights, saved optimizer, scheduler and random states will be loaded if exists. The default value is false.

39. _save_as_mlflow_model_ (bool, optional)

    Save as mlflow model with pyfunc as flavour. The default value is true.

# 2. Outputs
1. _output_dir_pytorch_ (custom_model, required)

    The folder containing finetuned model output with checkpoints, model config, tokenizer, optimzer and scheduler states and random number states in case of distributed training.

2. _output_dir_mlflow_ (URI_FOLDER, optional)

    Output dir to save the finetuned model as mlflow model.

# 4. Run Settings

This setting helps to choose the compute for running the component code. **For the purpose of finetune, gpu compute should be used**. We recommend using Standard_NC6s or Standard_NC6s_v3 compute.

> Select *Use other compute target*

- Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and the corresponding instance / cluster created in your workspace.
- If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute. See the figure below
![other compute target](../../images/other_compute_target_for_image_components.png)

## 4.1. Settings for Distributed Training

> When creating the compute, set the `Maximum number of nodes` to the desired value for multi-node training as shown in the figure below

![set maximum nodes](../../images/maximum_num_nodes.png)

> In case of distributed training, a.k.a multi-node training, the mode must be set to `Mount` and not `Upload` as shown in the figure below

![Output settings finetune](../../images/image_classification_output_settings.png)

> Set the number of processes under Distribution subsection to use all the gpus in a node

To use all the gpus within a node, set the `Process count per instance` to number of gpus in that node as shown below

![process count per instance](../../images/process_count_per_instance.png)

> Set the number of nodes under the Resources subsection

In case of distributed training, you can configure `instance count` under this subsection to increase the number of nodes as shown below

![instance count](../../images/instance_count.png)