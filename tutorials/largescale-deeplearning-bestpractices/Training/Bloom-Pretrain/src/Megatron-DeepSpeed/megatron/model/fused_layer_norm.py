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

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers


from megatron import get_args
from megatron import mpu
from packaging import version
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import importlib
import torch
import torch.nn.functional as F

global fused_mix_prec_layer_norm_cuda
fused_mix_prec_layer_norm_cuda = None


class FusedLayerNormAffineFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, weight, bias, normalized_shape, eps):

    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    input_ = input.contiguous()
    weight_ = weight.contiguous()
    bias_ = bias.contiguous()
    output, mean, invvar = fused_mix_prec_layer_norm_cuda.forward_affine(
        input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
    ctx.save_for_backward(input_, weight_, bias_, mean, invvar)

    return output


  @staticmethod
  def backward(ctx, grad_output):

    input_, weight_, bias_, mean, invvar = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    grad_input, grad_weight, grad_bias \
      = fused_mix_prec_layer_norm_cuda.backward_affine(
        grad_output.contiguous(), mean, invvar,
        input_, ctx.normalized_shape,
        weight_, bias_, ctx.eps)

    return grad_input, grad_weight, grad_bias, None, None



class MixedFusedLayerNorm(torch.nn.Module):

  def __init__(self, normalized_shape, eps=1e-5):
    super(MixedFusedLayerNorm, self).__init__()

    global fused_mix_prec_layer_norm_cuda
    fused_mix_prec_layer_norm_cuda = importlib.import_module(
      "fused_mix_prec_layer_norm_cuda")

    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    self.normalized_shape = torch.Size(normalized_shape)
    self.eps = eps
    self.weight = Parameter(torch.Tensor(*normalized_shape))
    self.bias = Parameter(torch.Tensor(*normalized_shape))
    self.reset_parameters()

    args = get_args()
    self.layernorm_tp_auto_sync = args.sync_tp_duplicated_parameters

    self.use_meg_ds_fused_layer_norm = (
      args.bf16 # Current Meg-DS cuda kernel has better throughput than torch.nn.LayerNorm
      or version.parse(torch.__version__) >= version.parse("1.11.0") # https://github.com/pytorch/pytorch/pull/66920
    )


  def reset_parameters(self):

    init.ones_(self.weight)
    init.zeros_(self.bias)


  def forward(self, input):

    if self.layernorm_tp_auto_sync:
      torch.distributed.all_reduce(self.weight, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())
      torch.distributed.all_reduce(self.bias, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())

    if self.use_meg_ds_fused_layer_norm:
        return FusedLayerNormAffineFunction.apply(
            input, self.weight, self.bias, self.normalized_shape, self.eps)
    else:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias)
