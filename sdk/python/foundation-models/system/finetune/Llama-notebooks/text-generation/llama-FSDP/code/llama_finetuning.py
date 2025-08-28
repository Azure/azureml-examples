# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from dataclasses import asdict
import fire
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
import torch.distributed as dist
import gc

# Unused imports removed
from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    print_model_size,
    get_policies,
    load_llama_model,
    predict,
    cleanup,
)

from utils.dataset_utils import get_preprocessed_dataset
from utils.train_utils import clear_gpu_cache
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.utils.data import DistributedSampler
import policies
from policies import AnyPrecisionAdamW
from configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main(**kwargs):
    # Update the configuration for the training and sharding process
    print("*****************Command line Arguments*******************")
    for k, v in kwargs.items():
        print(f"{k} = {v}")
    update_config((train_config, fsdp_config), **kwargs)
    train_config.task_name = kwargs["task_name"]
    print("*****************Training Configuration************************")
    for key in train_config.__annotations__.keys():
        print(f"{key} = {getattr(train_config, key)}")

    print("****************FSDP Configuration********************")
    for key in fsdp_config.__annotations__.keys():
        print(f"{key} = {getattr(fsdp_config, key)}")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(local_rank)

    # Calculate gradient accumulation steps
    # gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size
    gradient_accumulation_steps = 1

    dataset_config = generate_dataset_config(train_config, kwargs)
    print("****************Dataset Configuration********************")
    for key in dataset_config.__annotations__.keys():
        print(f"{key} = {getattr(dataset_config, key)}")

    print(f"Loading the model for process {os.environ['LOCAL_RANK']}")
    # Load the pre-trained model and setup its configuration

    if train_config.task_name == "text-classification":
        # For Text-classification
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = LlamaForSequenceClassification.from_pretrained(
                    train_config.model_name,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    num_labels=dataset_config.num_labels,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(
                    train_config.model_name, num_labels=dataset_config.num_labels
                )
                with torch.device("meta"):
                    model = LlamaForSequenceClassification(llama_config)
        else:
            model = LlamaForSequenceClassification.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                num_labels=dataset_config.num_labels,
            )
    else:
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = LlamaForCausalLM.from_pretrained(
                    train_config.model_name,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(train_config.model_name)
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)
        else:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )

    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer

            model = BetterTransformer.transform(model)
        except ImportError:
            print(
                "Module 'optimum' not found. Please install 'optimum' it before proceeding."
            )
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(
        os.path.join(train_config.model_name, "../tokenizer"), legacy=False
    )
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    tokenizer.pad_token = tokenizer.eos_token
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        print("****************PEFT Configuration********************")
        for key in peft_config.__annotations__.keys():
            print(f"{key} = {getattr(peft_config, key)}")
        for key, val in peft_config.__dict__.items():
            if key[:2] != "__" and not callable(v):
                print(f"{key} = {val}")

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy
            if train_config.use_peft
            else wrapping_policy,
            mixed_precision=mixed_precision_policy
            if not fsdp_config.pure_bf16
            else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            cpu_offload=CPUOffload(offload_params=True)
            if fsdp_config.cpu_offload
            else None,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(
                device=torch.device("cuda"), recurse=False
            )
            if train_config.low_cpu_fsdp and rank != 0
            else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=False,
            )

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=False,
        collate_fn=default_data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=False,
            collate_fn=default_data_collator,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]

    print("*****************Training Completed*******************")
    # End the distributed process communication
    cleanup()
    del model
    gc.collect()
    clear_gpu_cache()

    if rank == 0:
        # if True:
        # Generate predictions works for text-generation task only
        if (
            train_config.generate_predictions
            and train_config.task_name == "text-generation"
        ):
            prediction_dataset = get_preprocessed_dataset(
                tokenizer,
                dataset_config,
                split=dataset_config.prediction_split,
            )
            pred_dataloader = torch.utils.data.DataLoader(
                prediction_dataset,
                batch_size=1,
                num_workers=train_config.num_workers_dataloader,
                pin_memory=True,
                sampler=None,
                drop_last=False,
                collate_fn=default_data_collator,
            )
            model = load_llama_model(train_config, fsdp_config)
            predict(
                local_rank,
                model,
                pred_dataloader,
                tokenizer,
                dataset_config.max_gen_length,
                train_config.artifacts_dir,
                train_config.task_name,
            )


if __name__ == "__main__":
    fire.Fire(main)
