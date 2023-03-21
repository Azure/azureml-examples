# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron initialization."""

import random
import os
import sys
import time

import numpy as np
import torch
import logging as lg
import subprocess

from megatron import fused_kernels, logging
from megatron import get_adlr_autoresume
from megatron import get_args
from megatron import get_tensorboard_writer
from megatron import mpu
from megatron.global_vars import set_global_variables
from megatron.mpu import (set_tensor_model_parallel_rank,
                          set_tensor_model_parallel_world_size)

import deepspeed


def git_ds_info():
    args = get_args()
    if not args.deepspeed:
        return

    from deepspeed.env_report import main as ds_report
    ds_report(hide_operator_status=True, hide_errors_and_warnings=True)

    def command_exists(cmd):
        result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
        return result.wait() == 0

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD  2>&1"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD 2>&1"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"

    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')


def initialize_megatron(extra_args_provider=None, args_defaults={},
                        ignore_unknown_args=False, allow_no_cuda=False):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), 'Megatron requires CUDA.'

    # Parse args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(extra_args_provider=extra_args_provider,
                         args_defaults=args_defaults,
                         ignore_unknown_args=ignore_unknown_args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print('> setting random seeds to {} ...'.format(args.seed))

        def set_verbosity(logging_level: str):
            log_level = logging.log_levels[logging_level]
            logging.set_verbosity(log_level)
            logging.disable_default_handler()
            handler = lg.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            handler.flush = sys.stderr.flush
            logging.add_handler(handler)

        def set_verbosity_deepspeed(logging_level: str):
            if not args.deepspeed:
                return
            from deepspeed.utils import logger as ds_logger
            log_level = logging.log_levels[logging_level]
            ds_logger.setLevel(log_level)

        def set_verbosity_transformers(logging_level: str):
            try:
                # XXX: perhaps we need a better way of knowing when to override transformers logging
                # currently it's only when using `--tokenizer-type PretrainedFromHF`
                from transformers.utils import logging as transformers_logging
                log_level = logging.log_levels[logging_level]
                logging.set_verbosity(log_level)
            except:
                pass

        if args.rank == 0:
            if args.log_level is not None:
                set_verbosity(args.log_level)
                set_verbosity_deepspeed(args.log_level)
                set_verbosity_transformers(args.log_level)
        else:
            if args.log_level_replica is not None:
                set_verbosity(args.log_level_replica)
                set_verbosity_deepspeed(args.log_level_replica)
                set_verbosity_transformers(args.log_level_replica)
        _set_random_seed(args.seed)

    args = get_args()

    if args.rank == 0:
        git_ds_info()

    if  args.lazy_mpu_init:
        args.use_cpu_initialization=True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Initialize memory buffers.
        _initialize_mem_buffs()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        # No continuation function
        return None


def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.data.dataset_utils import compile_helper
        compile_helper()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = \
        (args.num_attention_heads / args.tensor_model_parallel_size) * \
        args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = seq_len > 16 and seq_len <=2048 and \
        seq_len % 4 == 0 and attn_batch_size % 4 == 0
    # Print a warning.
    if not ((args.fp16 or args.bf16) and
            custom_kernel_constraint and
            args.masked_softmax_fusion):
        if args.rank == 0:
            error = "constraints for invoking optimized fused softmax kernel are not met"
            if args.abort_on_unmet_fused_kernel_constraints:
                sys.exit(f"\n\nERROR: {error} and --abort-on-unmet-fused-kernel-constraints was passed. Aborting.\n\n")
            else:
                print(f'WARNING: {error}. We default back to unfused kernel invocations.', flush=True)

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling and loading fused kernels ...', flush=True)
        fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        import warnings
        with warnings.catch_warnings():
            # ignore loading noise
            warnings.simplefilter("ignore")
            fused_kernels.load(args)

    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>>> done with compiling and loading fused kernels. '
              'Compilation time: {:.3f} seconds'.format(
                  time.time() - start_time), flush=True)


def setup_deepspeed_random_and_activation_checkpointing(args):
    '''Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.
    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.
    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    '''
    num_layers = args.num_layers // args.checkpoint_num_layers
    num_layers = num_layers if args.num_layers % args.checkpoint_num_layers == 0 else num_layers + 1
    if args.split_transformers:
        num_layers *= 2

    deepspeed.checkpointing.configure(
        mpu,
        partition_activations=args.partition_activations,
        contiguous_checkpointing=args.contigious_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=args.checkpoint_in_cpu,
        synchronize=args.synchronize_each_layer,
        profile=args.profile_backward)

    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
        # Call the init process
        deepspeed.init_distributed(args.distributed_backend)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            mpu.initialize_model_parallel(args.tensor_model_parallel_size,
                                          args.pipeline_model_parallel_size,
                                          args.virtual_pipeline_model_parallel_size)

    if args.deepspeed and args.deepspeed_activation_checkpointing:
        setup_deepspeed_random_and_activation_checkpointing(args)


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)),
                            global_step=args.iteration)


def log_restart_to_tensorboard():
    """
    Log new start and world size - the key is to denote a restart, and use world_size as another
    useful info which can help to track changes in resource allocation.
    """
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        # emulate a blip to avoid flatline
        writer.add_scalar('iteration-time/world_size', args.world_size, args.iteration)
        writer.add_scalar('iteration-time/world_size', 0, args.iteration+1)


def _initialize_mem_buffs():
    """Initialize manually allocated static memory."""
    args = get_args()

    # Initialize memory for checkpointed activations.
    if args.distribute_checkpointed_activations:
        mpu.init_checkpointed_activations_memory_buffer()
