# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
import torch
from tqdm import tqdm

from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
from pkg_resources import packaging
from .memory_utils import MemoryTrace
import model_checkpointing
import torch.cuda.nccl as nccl
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from policies import fpSixteen, bfSixteen_mixed, get_llama_wrapper
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from model_checkpointing import (
    save_model_checkpoint,
    save_model_and_optimizer_sharded,
    save_optimizer_checkpoint,
)
from azureml.metrics import compute_metrics, constants
import numpy as np
import mlflow
import json
import time
from transformers import (
    LlamaForCausalLM,
)
from peft import PeftModel
import datetime


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    fsdp_config=None,
    local_rank=None,
    rank=None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []
    val_acc = 0.0
    results = {}
    eval_metrics = None
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            data_set_len = 0

            for step, batch in enumerate(
                tqdm(train_dataloader, colour="blue", desc=f"Training Epoch{epoch}")
            ):
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to("cuda:0")

                loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                first_key = next(iter(batch))
                data_set_len += len(batch[first_key])
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        optimizer.step()
                        optimizer.zero_grad()

                # print(f"\n step {step} is completed and loss is {loss.detach().float()}")
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        train_epoch_loss = total_loss / train_dataloader.dataset.num_rows
        if train_config.enable_fsdp:
            world_size = int(os.environ["WORLD_SIZE"])
            train_epoch_loss = train_epoch_loss / world_size

        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        print(f"Max CUDA memory allocated was {memtrace.peak} GB")
        print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
        print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
        print(
            f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
        )

        # Update the learning rate as needed
        lr_scheduler.step()

        eval_epoch_loss = 0.0
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, eval_metrics = evaluation(
                model, train_config, eval_dataloader, local_rank, rank, tokenizer
            )
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                if train_config.enable_fsdp:
                    dist.barrier()
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        print(f"we are about to save the PEFT modules")
                        save_policy = FullStateDictConfig(
                            offload_to_cpu=True, rank0_only=True
                        )
                        with FSDP.state_dict_type(
                            model, StateDictType.FULL_STATE_DICT, save_policy
                        ):
                            cpu_state = model.state_dict()
                        if rank == 0:
                            model.save_pretrained(
                                train_config.output_dir, state_dict=cpu_state
                            )
                            # save_name = "full_model_weights.pt"
                            # torch.save(cpu_state, os.path.join(train_config.output_dir, save_name))
                            print(
                                f"PEFT modules are saved in {train_config.output_dir} directory"
                            )
                    else:
                        print(f"we are about to save the PEFT modules")
                        model.save_pretrained(train_config.output_dir)
                        print(
                            f"PEFT modules are saved in {train_config.output_dir} directory"
                        )

                else:
                    if (
                        not train_config.use_peft
                        and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT
                    ):

                        save_model_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                    elif (
                        not train_config.use_peft
                        and fsdp_config.checkpoint_type
                        == StateDictType.SHARDED_STATE_DICT
                    ):
                        print(
                            " Saving the FSDP model checkpoints using SHARDED_STATE_DICT"
                        )
                        print("=====================================================")

                        save_model_and_optimizer_sharded(model, rank, train_config)
                        if train_config.save_optimizer:
                            save_model_and_optimizer_sharded(
                                model, rank, train_config, optim=optimizer
                            )
                            print(
                                " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT"
                            )
                            print(
                                "====================================================="
                            )

                    if not train_config.use_peft and train_config.save_optimizer:
                        save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                        print(
                            " Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT"
                        )
                        print("=====================================================")
                if train_config.enable_fsdp:
                    dist.barrier()

            if rank == 0 and eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
            if eval_metrics and eval_metrics["accuracy"] > val_acc:
                val_acc = eval_metrics["accuracy"]
                print(f"best eval accuracy on epoch {epoch} is {val_acc}")
            val_loss.append(best_val_loss)
            val_prep.append(eval_ppl)

        if rank == 0:
            print(
                f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}"
            )

            # Log to azureml dashboard
            mlflow.log_metric("train_loss", train_epoch_loss)
            mlflow.log_metric("eval_loss", eval_epoch_loss)
            if eval_metrics is not None:
                mlflow.log_metric("eval_accuracy", eval_metrics["accuracy"])

        lr_scheduler.step()

    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
        results["eval_accuracy"] = val_acc

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.barrier()
    model.eval()
    eval_outs = []
    eval_labels = []
    eval_loss = 0.0  # Initialize evaluation loss
    try:
        # Get the number of classes in the dataset. Only applicable to hugging face datasets
        num_labels = len(eval_dataloader.dataset.features["label"].names)
    except Exception as e:
        num_labels = -1
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
            tqdm(eval_dataloader, colour="green", desc="evaluating Epoch")
        ):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to("cuda:0")
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()

                if train_config.task_name == constants.Tasks.TEXT_CLASSIFICATION:
                    preds = torch.argmax(outputs.logits, -1)
                    eval_outs.extend(preds)
                    eval_labels.extend(batch["labels"])

    eval_outs = (
        torch.stack(eval_outs) if eval_outs else torch.tensor(eval_outs, device="cuda")
    )
    eval_labels = (
        torch.stack(eval_labels)
        if eval_labels
        else torch.tensor(eval_labels, device="cuda")
    )
    eval_metrics = None

    if train_config.task_name == constants.Tasks.TEXT_CLASSIFICATION:
        if rank == 0:
            eval_dataset_len = eval_dataloader.dataset.num_rows
            all_outputs = [torch.zeros_like(eval_outs) for i in range(world_size)]
            all_labels = [torch.zeros_like(eval_labels) for i in range(world_size)]
            dist.gather(eval_outs, all_outputs, 0)
            dist.gather(eval_labels, all_labels, 0)

            all_preds_cpu = [
                convert_tensor_to_primitives(pred.detach().cpu())
                for outputs in all_outputs
                for pred in outputs
            ][:eval_dataset_len]
            all_labels_cpu = [
                convert_tensor_to_primitives(label.detach().cpu())
                for labels in all_labels
                for label in labels
            ][:eval_dataset_len]

            eval_metrics = compute_additional_metrics(
                all_preds_cpu,
                all_labels_cpu,
                train_config.task_name,
                tokenizer,
                num_labels,
            )
        else:
            dist.gather(eval_outs, dst=0)
            dist.gather(eval_labels, dst=0)

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.barrier()
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if rank == 0:
        if eval_metrics is not None:
            print(
                f" {eval_ppl=} {eval_epoch_loss=} eval accuracy = {eval_metrics['accuracy']}"
            )
        else:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    return eval_ppl, eval_epoch_loss, eval_metrics


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


def convert_tensor_to_primitives(val):
    """Convert from tensors to primitive types(list / scalar)"""
    if val.ndim > 0:
        converted_val = val.tolist()
    else:
        converted_val = val.item()
    return converted_val


def compute_additional_metrics(predictions, labels, task_type, tokenizer, num_labels=0):
    """Compute additional metrics using azureml-metrics package.
        Predictions and labels are converted to primitive types before computing metrics
    Predictions: list of predictions
    type Predictions: list
    Labels: list of labels
    type Labels: list
    task_type: task type. One of the values in [Text-classification, Text-generation, ...]
    type task_type: str
    num_labels: number of labels in the dataset. This is only required for text-classification task.
    type num_labels: int
    """
    additional_params = (
        {"class_labels": np.array(range(num_labels))} if num_labels > 0 else {}
    )
    if task_type == constants.Tasks.TEXT_CLASSIFICATION:
        outputs = predictions
    else:
        outputs = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = np.expand_dims(labels, axis=-1).tolist()

    metrics = compute_metrics(
        task_type=task_type, y_test=labels, y_pred=outputs, **additional_params
    )["metrics"]
    print(f"metrics={metrics}")
    return metrics


def load_llama_model(train_config, fsdp_config):
    """Load the llama model from the train_config.output_dir folder.
    :train_config: The Training configuration
    :type train_config: dataclasses.dataclass
    :fsdp_config: The fsdp configuration
    :type fsdp_config: dataclasses.dataclass
    """
    if fsdp_config.pure_bf16:
        dtype = torch.bfloat16
    elif fsdp_config.use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto",
        torch_dtype=dtype,
    )
    print(f"Loading the model from {train_config.output_dir}")
    peft_model = PeftModel.from_pretrained(model, train_config.output_dir)
    print("Model loaded")
    model = peft_model.merge_and_unload()
    return model


def predict(
    rank, model, pred_dataloader, tokenizer, max_gen_length, artifacts_folder, task_type
):
    """Predict on the given dataset. This also calculates the additional metrics if labels are provided
    :rank: rank of the current node in a distributed setting
    :type rank: int
    :model: The model to evaluate
    :type model: torch.nn.Module
    :pred_dataloader: The dataloader containing the evaluation data
    :type pred_dataloader: torch.utils.data.DataLoader
    :tokenizer: The tokenizer used to decode predictions
    :type tokenizer: transformers.PreTrainedTokenizer
    :max_gen_length: The maximum length of the generated text
    :type max_gen_length: int
    :artifacts_folder: The folder to save the predictions
    :type artifacts_folder: str
    :task_type: The task type. One of the values in [Text-classification, Text-generation, ...]
    :type task_type: str
    """
    print("*** Predict ***")
    model.eval()
    device = f"cuda:{rank}"

    all_predicted_texts = []
    generated_tokens = []
    labels = []
    start_time = time.time()
    for step, batch in enumerate(
        tqdm(pred_dataloader, colour="blue", desc=f"Generating Predictions")
    ):
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=max_gen_length,
                pad_token_id=tokenizer.pad_token_id,
            )
            pred_texts = tokenizer.batch_decode(
                gen_tokens.cpu().numpy(), skip_special_tokens=True
            )
            for gen_token in gen_tokens.cpu().numpy():
                generated_tokens.append(gen_token)
            if batch.get("labels", None) is not None:
                for label in batch["labels"].cpu().numpy():
                    labels.append(label)
            for pred_text in pred_texts:
                summary = pred_text.split("Summary:")[-1].strip()
                all_predicted_texts.append({"predicted_text": summary})

    os.makedirs(artifacts_folder, exist_ok=True)
    filename = "generated_text.jsonl"
    with open(os.path.join(artifacts_folder, filename), "w") as f:
        for entry in all_predicted_texts:
            json.dump(entry, f)
            f.write("\n")
    save_location = os.path.join("outputs", "generated_text.jsonl")
    with open(save_location, "w") as f:
        for entry in all_predicted_texts:
            json.dump(entry, f)
            f.write("\n")
    print(
        "Time taken to infer on samples = {} seconds".format(time.time() - start_time)
    )
    if len(labels) > 0:
        print("computing additional metrics")
        compute_additional_metrics(generated_tokens, labels, task_type, tokenizer)
