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

"""General utilities."""

import os
import sys
import warnings
from random import randint

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as torchDDP

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import get_args, logging
from megatron import print_rank_0
from megatron import get_adlr_autoresume
from megatron import mpu
from megatron.model.module import param_is_not_shared
from megatron.model.utils import log_debug_usage
from megatron.mpu.layers import param_is_not_tensor_parallel_duplicate, VocabParallelEmbedding
from megatron import get_num_microbatches

logger = logging.get_logger(__name__)

def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.cuda.IntTensor([0])
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, lr_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(
        data,
        eod_token,
        reset_position_ids,
        reset_attention_mask,
        eod_mask_loss,
        prefix_indices,
        loss_on_targets_only,
    ):
    """
    Build masks and position id for left to right model.
    :param prefix_indices: argument can have multiple types:
        - None signifies that the model is fully autoregressive.
        - List[int] the argument holds all prefix indices that split a row into an input and a target
        - List[List[int]] the argument holds all prefix indices that split documents between input and target.
    :param loss_on_targets_only: bool to determine if we should mask loss on prefix.
    """

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask or prefix_indices is not None:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask or prefix_indices is not None:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]

            # If the last eod token is not the last token of the sequence, we suppose that there is a partial document
            # We treat this case as if we add an eod token at the end of the sequence.
            if data[b][-1] != eod_token:
                eod_index = torch.cat(
                    (eod_index, torch.tensor([len(data[b])], dtype=eod_index.dtype, device=eod_index.device))
                )

            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]

                if reset_attention_mask:
                    # Prevent cross document interactions.
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0

                    # Prefix lm per document.
                    if prefix_indices:
                        assert isinstance(prefix_indices[b], list), f"prefix for a row has to be document specific, and consequently return a list, got {prefix_indices[b]}"
                        attention_mask[b, 0, prev_index: prefix_indices[b][j], prev_index: prefix_indices[b][j]] = 1
                        if loss_on_targets_only:
                            # Last token of the prefix should predict the prefix_index id
                            loss_mask[b, prev_index: prefix_indices[b][j] - 1] = 0.0

                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)

                prev_index = i + 1

            # Prefix lm per row.
            if prefix_indices is not None and (reset_attention_mask is False):
                assert isinstance(prefix_indices[b], int), \
                    f"prefix for a row has to be row specific, and consequently return an int, got {prefix_indices[b]}"
                attention_mask[b, 0, :prefix_indices[b], :prefix_indices[b]] = 1
                if loss_on_targets_only:
                    # Last token of the prefix should predict the prefix_index id
                    loss_mask[b, :prefix_indices[b] - 1] = 0.0

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_packed_attention_mask(is_causal: bool, causal_mask: torch.Tensor, decoder_is_inputs: torch.Tensor, segment_ids: torch.Tensor):
    """
    Inspired by https://github.com/google-research/t5x/blob/7193407f98a8b18100b71a04ff777238be1682ca/t5x/examples/decoder_only/layers.py#L978

    Arguments:
        - is_causal: determines if the masking should be causal in the `inputs` part
        - causal_mask: torch.BoolTensor [batch_size, sequence_length, sequence_length]
        - decoder_is_inputs: torch.BoolTensor [batch_size, sequence_length]
        - segment_ids: torch.IntTensor [batch_size, sequence_length]
    Returns:
        - attention_mask: torch.BoolTensor [batch_size, 1, sequence_length, sequence_length]
    """

    """Causal Inputs Mask:
    mask = [[[[1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1]]]]
    """
    assert causal_mask.dtype == torch.bool
    assert segment_ids.dtype == torch.long
    if is_causal:
        causal_inputs_mask = causal_mask
    else:
        assert decoder_is_inputs.dtype == torch.bool
        inputs_mask = decoder_is_inputs[:, None, :, None] * decoder_is_inputs[:, None, None, :]
        causal_inputs_mask = causal_mask + inputs_mask

    """Padding Mask:
    mask = [[[[1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]]]]
    """
    padding_mask = (segment_ids != 0)[:, None, :, None] * (segment_ids != 0)[:, None, None, :]

    """Segment Mask:
    mask = [[[[1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]]]]
    """
    segment_mask = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]

    """Final Mask:
    mask = [[[[1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]]]]
    """
    attention_mask = causal_inputs_mask * padding_mask * segment_mask

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask

def param_size(parameter):
    return parameter.ds_numel if hasattr(parameter, 'ds_id') else parameter.nelement()


def unique_param_count(param_list):
    # not actually deduplicating tied variables for now (which causes the PP > 1 double-counting bug)
    return sum(dict((p.data_ptr(), param_size(p)) for p in param_list).values())


def non_embedding_params(module):
    embedding_param_names = [
        f"{name}.weight" for name, module_type in module.named_modules() if isinstance(module_type, nn.Embedding) or isinstance(module_type, VocabParallelEmbedding)
    ]
    non_embedding_parameters = [
        parameter for name, parameter in module.named_parameters() if name not in embedding_param_names
    ]
    return unique_param_count(non_embedding_parameters)


def get_parameters_in_billions(model, exclude_embeddings=False):
    gpus_per_model = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())

    if exclude_embeddings:
        approx_parameters_in_billions = sum([non_embedding_params(model_module) for model_module in model])
    else:
        args = get_args()
        if args.rank == 0:
            warnings.warn("Parameter count with the embeddings will be inaccurate with PP > 1, as the first and last stage hold several copies of the embeddings")
        approx_parameters_in_billions = unique_param_count([p for model_module in model for p in model_module.parameters()])

    return approx_parameters_in_billions*gpus_per_model/(1e9)


def flops_calculator(model, args, iteration_time):
    return # currently broken
    gpus_per_model = torch.distributed.get_world_size(group = mpu.get_model_parallel_group())

    approx_parameters_in_billions = get_parameters_in_billions(model)

    batch_size = args.micro_batch_size * get_num_microbatches()

    giga_flops_per_model_per_train_step = approx_parameters_in_billions * batch_size * args.seq_length * 2.0 * 4.0

    effective_tera_flops_per_gpu = giga_flops_per_model_per_train_step / (iteration_time * 1000.0 * gpus_per_model)

    print_rank_0(f"Effective Tera Flops per GPU: {round(effective_tera_flops_per_gpu, 2)} and total parameters {round(approx_parameters_in_billions, 3)} B")

def get_prefix_indices(data, eod_token, partial_prefix_indices, reset_attention_mask):
    """
    Helper function in order to:
     - randomly choose prefix index when there's no constraint
     - check that prefix are compatible with convention.

    :param data: torch.Tensor
    :param eod_token: int, token_id used to signal end of document
    :param partial_prefix_indices: this agument can have multiple types:
        - None, it signals that all prefix indices are randomly sampled.
        - List[Optional[int]], its length has to be equal to mini batch size. It stores all the indices for per row prefix.
            Optional means that if set to None, we allows ourselves to sample one randomly.
        - List[List[Optional[int]]], it follows the following rules:
            - The first dimension refers to that sample, ie len(partial_prefix_indices) == len(data)
            - The second dimension refers to the number of document of that sample, ie
                len(partial_prefix_indices[b]) == (data[b] == eod_token).sum() (+1 for the last partial document).
            - partial_prefix_indices have to be interleaved with eod_indices, ie
                eod_indices[b][d-1] < partial_prefix_indices[b][d] < eod_indices[b][d] + 1 or is None.
            - Optional means that if set to None, we allows ourselves to sample one randomly.
    :param reset_attention_mask: bool, determines if prefixes are to be per document or per row.
    :return Depending if prefix is per document or per row, the method returns:
        - List[List[int]]: prefix indices for each document in case of per document prefix
        - List[int]: prefix indices for rows else.
    """
    micro_batch_size, seq_length = data.size()
    prefix_indices = []

    assert partial_prefix_indices is None or len(partial_prefix_indices) == micro_batch_size, f"partial_prefix_indices has to be None or its length equal to {micro_batch_size}, got {len(partial_prefix_indices)}"
    for batch_id in range(micro_batch_size):
        # Prefix lm per document.
        if reset_attention_mask:
            prefix_indices.append([])

            # Compute the index of all eod tokens in data.
            eod_indices = (data[batch_id] == eod_token).nonzero().squeeze(-1)

            # If the last eod token is not the last token of the sequence, we suppose that there is a partial document
            # We treat this case as if we add an eod token at the end of the sequence.
            if data[batch_id][-1] != eod_token:
                eod_indices = torch.cat(
                    (eod_indices,
                     torch.tensor([len(data[batch_id])], dtype=eod_indices.dtype, device=eod_indices.device))
                )

            prev_index = 0
            assert partial_prefix_indices is None or len(partial_prefix_indices[batch_id]) == len(eod_indices), f"The number of prefixes has to match the number of documents, complete or partial. Got {len(partial_prefix_indices[batch_id])} prefixes and {len(eod_indices)} documents"

            for doc_id, eod_index in enumerate(eod_indices):
                assert partial_prefix_indices is None or isinstance(partial_prefix_indices[batch_id], list), f"Per document prefix has to store a list on indices for each row, got {partial_prefix_indices[batch_id]}"
                # Prefix index is defined as the first index that isn't attended by all tokens in a document
                if partial_prefix_indices is None or partial_prefix_indices[batch_id][doc_id] is None:
                    # We need to randomly generate a prefix index that satisfies the interleave condition in the docstring
                    prefix_index = randint(prev_index + 1, eod_index)
                else:
                    # We get value from partial_prefix_indices, and run validation on that value
                    prefix_index = partial_prefix_indices[batch_id][doc_id]
                assert prev_index + 1 <= prefix_index <= eod_index, f"Prefix index needs to be between documents indices, {prev_index + 1} <= {prefix_index} <= {eod_index} should be True."

                prefix_indices[batch_id].append(prefix_index)
                prev_index = eod_index + 1

        # Prefix lm per row.
        else:
            assert partial_prefix_indices is None or isinstance(partial_prefix_indices[batch_id], int), \
                f"Per document prefix has to store an int for each row, got {partial_prefix_indices[batch_id]}"

            # Prefix index is defined as the first index that isn't attended by all previous tokens in a document
            prefix_index: int
            if partial_prefix_indices is None or partial_prefix_indices[batch_id] is None:
                # 0 being the first prefix index makes no sense since 0 always attends to itself, and there are no other tokens before.
                prefix_index = randint(1, seq_length)
            else:
                # We get value from partial_prefix_indices, and run validation on that value
                prefix_index = partial_prefix_indices[batch_id]
            assert 1 <= prefix_index <= seq_length, f"Prefix index needs to be between documents indices, 1 <= {prefix_index} <= {seq_length} should be True."
            prefix_indices.append(prefix_index)

    return prefix_indices


@log_debug_usage(logger, "Using loss reweighting")
def reweight_loss_mask_(loss_mask: torch.Tensor, tokens: torch.Tensor):
    """Reweight loss mask in-place"""
    _, seq_length = tokens.shape
    weight_loss = torch.arange(seq_length, 0, -1, dtype=torch.float, device=loss_mask.device) / (seq_length + 1) * 2
    # in-place operation
    loss_mask *= weight_loss[None, :]


def found_kill_switch():
    args = get_args()
    if args.kill_switch_path is not None and os.path.exists(args.kill_switch_path):
        return True
    else:
        return False

def get_fingerprint_header():
    return f"{'min':^13} {'max':^13} {'mean':^13} {'l2 norm':^12} metadata"

def get_fingerprint(p):
    return f"{p.min():13.6e} {p.max():13.6e} {p.mean():13.6e} {p.norm():12.6e}"


def dump_weights(preamble, iteration, model, optimizer, tensor=None):   
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    fn = f"debug-bf16-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"

    # only care for first and last pp stages and dp0 tp0
    #if not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()):
    #    return

    #if not (tp_rank == 0 and dp_rank == 0):
    #    return

    if tensor is not None:
        orig_tensor = tensor
        if hasattr(tensor, "_hp_param"):
            numel = tensor._hp_param.numel() # // dp_size
            tensor = tensor.flatten().narrow(0, 0, numel)

    #print(fn)
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")

        if tensor is not None:
            fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
        else:
            for n, p in model[0].named_parameters():
                fh.write(f"{get_fingerprint(p)} {n} {p.shape}\n")


    return 


    # until we figure out how to dump the actual fp32 values don't do this
    fn = f"debug-fp32-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")
        if tensor is not None:
            tensor = orig_tensor
            if hasattr(tensor, "_hp_param"):
                fh.write(f"{get_fingerprint(tensor._hp_param)} tensor {tensor._hp_param.shape}\n")
                #fh.write(f"{get_fingerprint(tensor._hp_grad)} tensor grad\n")
            else:
                fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
                #fh.write(f"{get_fingerprint(tensor.grad)} tensor grad\n")

        else:
            if hasattr(model[0].module.tied_modules, "embed"):
                p = model[0].module.tied_modules.embed.word_embeddings.weight._hp_param
                fh.write(f"{get_fingerprint(p)} module.tied_modules.embed.word_embeddings.weight._hp_param {p.shape}\n")

        # for i, param_group in enumerate(optimizer.param_groups):
        #     fh.write(f"{get_fingerprint(optimizer.fp32_groups_flat_partition[i])} group={i}\n")
            #fh.write(f"{i}={optimizer.fp32_groups_flat_partition[i]}\n")
    #     if mpu.is_pipeline_first_stage():
    #         x = optimizer.fp32_groups_flat_partition[0]
    #         fh.write(f"fp32={x[:402432]}\n")
    #     if mpu.is_pipeline_last_stage()):
    #         x = optimizer.fp32_groups_flat_partition[1]
    #         fh.write(f"fp32={x[-402432:]}\n")

    # import os
    # import socket
    # hostname = socket.gethostname()
    # pid = os.getpid()
    # global_rank = torch.distributed.get_rank()
    #fn = f"debug-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-global{global_rank}-{preamble}-{pid}.txt"