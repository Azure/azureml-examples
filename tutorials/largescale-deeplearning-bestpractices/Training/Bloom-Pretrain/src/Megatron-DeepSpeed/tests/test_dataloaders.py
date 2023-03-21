import itertools
import os
import shutil
from typing import Set
from unittest.mock import patch

import deepspeed
import torch

import finetune_t0_non_causal_decoder
from megatron import global_vars, get_tokenizer, initialize_megatron, get_args
from megatron.data import mlm_dataset, mtf_dataset, decoder_packed_mtf_dataset
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.testing_utils import TestCasePlus, flatten_arguments, mockenv_context, torch_assert_equal


def get_default_args():
    """return a dictionary with key as argument name and value as additional arguments"""
    return {
        # GPT_ARGS
        "--num-layers": "2",
        "--hidden-size": "128",
        "--num-attention-heads": "4",
        "--seq-length": "512",
        "--max-position-embeddings": "512",
        "--micro-batch-size": "4",
        "--global-batch-size": "8",
        "--lr-decay-iters": "320000",
        "--lr-decay-style": "cosine",
        "--lr": "0.00015",
        "--min-lr": "1.0e-5",
        "--train-iters": "5000",
        "--tokenizer-type": "PretrainedFromHF",
        "--tokenizer-name-or-path": "gpt2",
        "--data-impl": "mmap",
        "--split": "949,50,1",
        "--distributed-backend": "nccl",
        "--weight-decay": "1e-2",
        "--clip-grad": "1.0",
        "--lr-warmup-fraction": ".01",
        "--fp16": "",

        "--attention-dropout": "0",
        "--hidden-dropout": "0",

        # OUTPUT_ARGS
        "--log-interval": "10",
        "--save-interval": "500",
        "--eval-interval": "100",
        "--eval-iters": "10",
        "--checkpoint-activations": "",

        # DATA_ARGS
    }

def get_dummy_mtf_decoder_packed_data(micro_batch_size: int, seq_length: int, vocab_size: int, special_tokens_ids: Set[int]):
    seq_length += 1

    num_segments = torch.randint(1, 5, ())
    segment_ids = torch.zeros(micro_batch_size, seq_length, dtype=torch.long)
    is_inputs = torch.zeros(micro_batch_size, seq_length, dtype=torch.bool)
    for batch_id in range(micro_batch_size):
        # - `*2`: Hack in order to two start_new_segements to be seperated with two tokens at least
        # - `+1`: Hack in order the start_mew_segments not to be 0
        start_new_segments = torch.sort(torch.randperm((seq_length - 2) // 2, )[:num_segments]).values * 2 + 1
        segment_ids[batch_id, start_new_segments] = 1

        end_inputs = [
            torch.randint(low=start_segment, high=end_segment, size=())
            for start_segment, end_segment in zip([0, *start_new_segments], [*start_new_segments, seq_length])
        ]
        for end_input, start_segment in zip(end_inputs, [0, *start_new_segments]):
            is_inputs[batch_id][start_segment: end_input + 1] = True

    segment_ids = torch.cumsum(segment_ids, dim=-1) + 1

    tokens = torch.randint(high=vocab_size, size=(micro_batch_size, seq_length), dtype=torch.long)
    flatten_token_view = tokens.view(-1,)
    for token_id in range(len(flatten_token_view)):
        token = flatten_token_view[token_id]
        # While token is a special tokens we change that token
        while token in special_tokens_ids:
            flatten_token_view[token_id] = (token + 1) % vocab_size
            token = flatten_token_view[token_id]

    return {
        "decoder_token_ids": tokens,
        "decoder_segment_ids": segment_ids,
        "decoder_is_inputs": is_inputs
    }

class TestDataLoading(TestCasePlus):
    def setUp(self) -> None:
        super().setUp()

        # We reset all global variables
        global_vars._GLOBAL_ARGS = None
        global_vars._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
        global_vars._GLOBAL_TOKENIZER = None
        global_vars._GLOBAL_TENSORBOARD_WRITER = None
        global_vars._GLOBAL_ADLR_AUTORESUME = None
        global_vars._GLOBAL_TIMERS = None

        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost", MASTER_PORT="9994", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"
        )

    def copy_data_to_temp(self, root_dir, prefix):
        """copy data to temp, and return paths to temp version"""
        src_path = os.path.join(root_dir, prefix)
        src_dirname = os.path.dirname(src_path)

        tmp_dir = self.get_auto_remove_tmp_dir()
        dest_path = os.path.join(tmp_dir, prefix)
        dest_dirname = os.path.dirname(dest_path)
        os.makedirs(dest_dirname, exist_ok=True)
        for folder in os.listdir(src_dirname):
            src_folder = os.path.join(src_dirname, folder)
            dest_folder = os.path.join(dest_dirname, folder)
            if src_folder.startswith(src_path):
                if os.path.isdir(src_folder):
                    shutil.copytree(src_folder, dest_folder)
                else:
                    shutil.copy2(src_folder, dest_folder)
        return dest_path

    def test_mlm_dataset(self):
        command_args = get_default_args()
        data_path = self.copy_data_to_temp(self.data_dir, "gpt2/meg-gpt2-openwebtext_text_document")
        command_args["--data-path"] = data_path
        command_args["--noise-density"] = "0.15"
        command_args["--mean-noise-span-length"] = "3"
        command_args["--vocab-extra-ids"] = "100"

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()

                # tokenizer
                tokenizer = get_tokenizer()
                # SEP is required to put in MLM preprocessed.
                tokenizer.tokenizer.add_special_tokens({"sep_token": "<s>"})

                args = get_args()
                train_val_test_num_samples = [
                    args.train_iters * args.global_batch_size,
                    args.eval_iters * args.global_batch_size,
                    0
                ]
                train_ds, valid_ds, test_ds = mlm_dataset.build_train_valid_test_datasets(
                    data_prefix=args.data_path,
                    data_impl=args.data_impl,
                    splits_string=args.split,
                    # TODO @thomasw21 figure how that value works
                    train_valid_test_num_samples=train_val_test_num_samples,
                    sequence_length=args.seq_length,
                    noise_density=args.noise_density,
                    mean_noise_span_length=args.mean_noise_span_length,
                    seed=args.seed,
                    skip_warmup=(not args.mmap_warmup)
                )

                sample = train_ds[0]
                # +1 is needed to compute labels. As inputs and targets are just concatenated.
                self.assertEqual(len(sample["input_tokens"]) + len(sample["target_tokens"]), args.seq_length + 1)

                # We make sure that inputs/targets end with <sep>
                self.assertEqual(sample["input_tokens"][-1], tokenizer.sep)
                self.assertEqual(sample["target_tokens"][-1], tokenizer.sep)

    def test_decoder_packed_mtf_dataloader(self):
        command_args = get_default_args()
        data_path = self.copy_data_to_temp(self.data_dir, "gpt2/ag_news_prompt")
        command_args["--data-path"] = data_path

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()

                args = get_args()
                tokenizer = get_tokenizer()
                # Hack: `gpt2` doesn't have a padding token, so we override that value.
                tokenizer.tokenizer.pad_token_id = tokenizer.tokenizer.eos_token_id

                train_val_test_num_samples = [
                    args.train_iters * args.global_batch_size,
                    args.eval_iters * args.global_batch_size,
                    0
                ]
                train_ds, valid_ds, test_ds = decoder_packed_mtf_dataset.build_train_valid_test_datasets(
                    data_prefix=args.data_path,
                    data_impl=args.data_impl,
                    splits_string=args.split,
                    # TODO @thomasw21 figure how that value works
                    train_valid_test_num_samples=train_val_test_num_samples,
                    seq_length=args.seq_length + 1,
                    pad_token=tokenizer.pad,
                    eos_token=tokenizer.eos,
                    seed=args.seed,
                    skip_warmup=(not args.mmap_warmup)
                )

                batch_iterator = build_pretraining_data_loader(
                    train_ds, consumed_samples=0, num_workers=4
                )

                last_padding_size = 0
                for i, items in enumerate(batch_iterator):
                    micro_batch_size, seq_length = items["decoder_token_ids"].shape

                    # Check dtypes
                    self.assertEqual(items["decoder_token_ids"].dtype, torch.int64)
                    self.assertEqual(items["decoder_segment_ids"].dtype, torch.int64)
                    self.assertEqual(items["decoder_is_inputs"].dtype, torch.bool)

                    # `micro_batch_size` correspond to the one in argument
                    self.assertEqual(micro_batch_size, args.micro_batch_size)
                    # `seq_length` correspond to the one in argument + 1 in order to get tokens/labels
                    self.assertEqual(seq_length, args.seq_length + 1)

                    original_samples_count = 0
                    for batch_id in range(micro_batch_size):
                        segment_ids = [k for k, _ in itertools.groupby(items["decoder_segment_ids"][batch_id])]
                        # `segment_ids` is [1,2,...]
                        self.assertEqual(segment_ids[:-1], list(range(1, len(segment_ids))))
                        # `0` signify that the tokens are padding
                        self.assertIn(segment_ids[-1], [0, len(segment_ids)])
                        original_samples_count += len([segment_id for segment_id in segment_ids if segment_id != 0])

                    # Test that we actually pack, ie we have more samples than the `batch_size`
                    self.assertGreater(original_samples_count, micro_batch_size)

                    # Test that the first sample of each batch couldn't fit inside the previous batch
                    first_sample_segment_ids = next(itertools.groupby(items["decoder_segment_ids"][0]))[1]
                    first_sample_size = len(list(first_sample_segment_ids))
                    self.assertGreater(first_sample_size, last_padding_size)

                    # update `last_padding_size`
                    last_padding_size = len([None for segment_id in items["decoder_segment_ids"][micro_batch_size - 1] if segment_id == 0])


    def test_finetune_t0_non_causal_decoder_get_batch_pipe(self):
        command_args = get_default_args()
        command_args["--position-embedding-type"] = "alibi"

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()

                args = get_args()
                tokenizer = get_tokenizer()
                # Hack: `gpt2` doesn't have a padding token, so we override that value.
                tokenizer.tokenizer.pad_token_id = tokenizer.tokenizer.eos_token_id

                # Dummy data
                data = get_dummy_mtf_decoder_packed_data(
                    micro_batch_size=args.micro_batch_size,
                    seq_length=args.seq_length,
                    vocab_size=args.padded_vocab_size,
                    special_tokens_ids={tokenizer.pad}
                )

                (tokens, position_ids, attention_mask), (labels, loss_mask) = finetune_t0_non_causal_decoder.get_batch_pipe(data)

                tokens = tokens.cpu()
                position_ids = position_ids.cpu()
                attention_mask = attention_mask.cpu()
                labels = labels.cpu()
                loss_mask = loss_mask.cpu()

                self.assertEqual(loss_mask.dtype, torch.float)
                torch_assert_equal(loss_mask.bool(), ~data["decoder_is_inputs"][:, 1:] * (data["decoder_token_ids"][:, :-1] != tokenizer.pad))
                torch_assert_equal(tokens, data["decoder_token_ids"][:, :-1])
                torch_assert_equal(labels, data["decoder_token_ids"][:, 1:])

                for batch_id in range(args.micro_batch_size):
                    segment_cuts = torch.nonzero(data["decoder_segment_ids"][batch_id, 1:] - data["decoder_segment_ids"][batch_id, :-1]) + 1
                    for segment_start, segment_end in zip([0, *segment_cuts], [*segment_cuts, args.seq_length]):
                        self.assertTrue(torch.all(attention_mask[batch_id, 0, segment_start: segment_end, :segment_start]))
                        self.assertTrue(torch.all(attention_mask[batch_id, 0, segment_start: segment_end, segment_end:]))

                # TODO @thomasw21 make sure that we reset `position_ids`
