# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    model_name: str = "/mnt/llama-models/Llama-2-70b/mlflow_model_folder/data/model"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    run_validation: bool = True
    generate_predictions: bool = True
    batch_size_training: int = 16
    num_epochs: int = 3
    num_workers_dataloader: int = 1
    lr: float = 4e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 16
    dataset = (
        "emotion_detection_dataset"  # "emotion_detection_dataset", "samsum_dataset"
    )
    peft_method: str = "lora"  # None , llama_adapter, prefix
    use_peft: bool = False
    output_dir: str = "/mnt/llama-models/Llama-2-13b/outputs"
    artifacts_dir: str = ""
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = False
    dist_checkpoint_root_folder: str = (
        "PATH/to/save/FSDP/model"  # will be used if using FSDP
    )
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    task_name: str = "text-generation"  # Will be used for calculation of metrics
    use_fast_kernels: bool = False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    max_new_tokens = 100  # Only used for text-generation tasks
