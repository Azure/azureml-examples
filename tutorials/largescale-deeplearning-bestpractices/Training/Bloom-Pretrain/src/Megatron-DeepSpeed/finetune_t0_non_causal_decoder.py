"""Multitask Finetuning T0"""

import torch

from megatron import get_args, get_tokenizer, print_rank_0, mpu
from megatron.data.decoder_packed_mtf_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.enums import PositionEmbeddingType, AttnMaskType
from megatron.model import GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_packed_attention_mask

import deepspeed
from deepspeed.runtime.utils import see_memory_usage

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        return fn

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building GPT model ...")
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == "none" else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed:
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                attn_mask_type=AttnMaskType.custom
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe
        else:
            raise NotImplementedError("DeepSpeed is required for T0")

    see_memory_usage(f"After Building Model", force=True)
    return model

def get_batch_pipe(data):
    """
    Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator` & in packed fashion
    
    data:
    decoder_tokens = [[6, 7, 8, 3, 4, 5, 0]]
    decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_is_inputs = [[1, 1, 0, 1, 1, 0, 0]]
    """
    args = get_args()
    tokenizer = get_tokenizer()

    # Broadcast data.
    data_b = mpu.broadcast_data(["decoder_token_ids", "decoder_segment_ids"], data, torch.int64)
    data_c = mpu.broadcast_data(["decoder_is_inputs"], data, torch.bool)

    # Unpack.
    tokens_ = data_b["decoder_token_ids"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    segment_ids = data_b["decoder_segment_ids"].long()[:, :-1]
    decoder_is_inputs = data_c["decoder_is_inputs"][:, :-1]

    # Get the masks and position ids.
    causal_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        prefix_indices=None,
        loss_on_targets_only=False # This is done below
    )
    # Only compute loss over causal target tokens, i.e. ignore input_tokens & padding
    loss_on_targets_only = ~data_c["decoder_is_inputs"][:, 1:]
    loss_on_non_pad_only = (tokens != tokenizer.pad)
    loss_mask *= loss_on_targets_only * loss_on_non_pad_only

    attention_mask = get_packed_attention_mask(
        # Run non-causal decoder
        is_causal=False,
        causal_mask=~(causal_mask.bool()),
        decoder_is_inputs=decoder_is_inputs.bool(),
        segment_ids=segment_ids.long(),
    )

    if args.position_embedding_type not in [PositionEmbeddingType.alibi, PositionEmbeddingType.rotary]:
        raise NotImplementedError("absolute positional embeddings require us to reset position_ids accordingly.")

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    tokenizer = get_tokenizer()

    print_rank_0("> building train, validation, and test datasets for T0 ...")
    # Option 1 of data loading using --data-path
    if args.data_path:
        # TODO: Not yet compatible with dataset weights (Will break at prefixes, weights = analyze_data_prefix(args.data_path))
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            seq_length=args.seq_length + 1,
            pad_token=tokenizer.pad,
            eos_token=tokenizer.eos,
            train_valid_test_num_samples=train_val_test_num_samples,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup)
        )
        # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
                              eval(f"args.{s}_weighted_split_weights"),
                              eval(f"args.{s}_weighted_split_splits"),
                              eval(f"args.{s}_weighted_split_names"))
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(
                    dataset_group_name=name,
                    paths=paths,
                    weights=weights,
                    splits=splits,
                    data_impl=args.data_impl,
                    train_valid_test_num_samples=train_val_test_num_samples,
                    seq_length=args.seq_length + 1,
                    pad_token=tokenizer.pad,
                    eos_token=tokenizer.eos,
                    seed=args.seed,
                    skip_warmup=(not args.mmap_warmup),
                    train_valid_test=s
                )
                eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating T0 datasets ...")
    return train_ds, valid_ds, test_ds

@record
def main():
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step_func=None,
        args_defaults={}
    )

if __name__ == "__main__":
    main()
