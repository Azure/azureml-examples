# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import ClassVar
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


@dataclass
class fsdp_config:
    mixed_precision: bool = True
    use_fp16: bool = False
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = (
        StateDictType.SHARDED_STATE_DICT
    )  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool = True
    pure_bf16: bool = True
    optimizer: str = "AdamW"
    cpu_offload: bool = False
