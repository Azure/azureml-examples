from random import randint
from typing import Set
from unittest.mock import patch

import deepspeed
import torch
from parameterized import parameterized
from torch import nn
import torch.nn.functional as F

from megatron.enums import AttnMaskType
from megatron.model.fused_layer_norm import MixedFusedLayerNorm
from packaging import version

from megatron import initialize_megatron, get_args, get_tokenizer, global_vars
from megatron.model.fused_softmax import ScaledMaskedSoftmax, FusedScaleMaskSoftmax
from megatron.model.utils import attention_mask_func
from megatron.testing_utils import TestCasePlus, mockenv_context, flatten_arguments, torch_assert_equal, \
    torch_assert_close, require_torch_bf16
from megatron.training import setup_model_and_optimizer
import pretrain_gpt
import pretrain_prefix_lm
import finetune_t0_non_causal_decoder


def get_default_args(test_file_dir: str):
    """return a dictionary with key as argument name and value as additional arguments"""
    return {
        # GPT_ARGS
        "--num-layers": "2",
        "--hidden-size": "128",
        "--num-attention-heads": "4",
        "--seq-length": "256",
        "--max-position-embeddings": "256",
        "--micro-batch-size": "2",
        "--global-batch-size": "2",
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
        "--inference": "",

        "--attention-dropout": "0",
        "--hidden-dropout": "0",

        # OUTPUT_ARGS
        "--log-interval": "10",
        "--save-interval": "500",
        "--eval-interval": "100",
        "--eval-iters": "10",
        "--checkpoint-activations": "",

        # DATA_ARGS

        # DeepSpeed args
        "--deepspeed": "",
        "--deepspeed_config": f"{test_file_dir}/ds_config_inference.json",
        "--zero-stage": "0",
    }


def equal_vectors(tensor1, tensor2, dim=-1):
    """View tensor1 and tensor2 as a list of vectors, and compute equality"""
    return torch.linalg.norm(tensor1 - tensor2, dim=dim) == 0


def iter_out_of_one(one):
    return iter([one])


def get_dummy_mtf_decoder_packed_data(micro_batch_size: int, seq_length: int, vocab_size: int, special_tokens_ids: Set[int]):
    """Code from `tests/test_dataloaders.py"""
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
            torch.randint(low=start_segment, high=end_segment - 1, size=())
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


class MyTestCase(TestCasePlus):
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

    def test_gpt(self):
        """Test causal invariance, ie past token don't depend on future tokens."""
        command_args = get_default_args(self.test_file_dir_str)

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()
                tokenizer = get_tokenizer()

                model, _, _ = setup_model_and_optimizer(pretrain_gpt.model_provider)
                model = model[0]
                model._config.train_micro_batch_size_per_gpu = args.micro_batch_size
                model.set_train_batch_size(args.micro_batch_size)

                token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

                # eod is a special token
                token_ids[token_ids == tokenizer.eod] += 1
                token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

                # get a modified version of the first batch, we change a specific index
                changed_index = randint(0, args.seq_length - 2)
                token_ids_changed = token_ids.clone()
                # We increment the token_id by one for that index in order to artificially change the sequence.
                token_ids_changed[:, changed_index] = \
                    (token_ids_changed[:, changed_index] + 1) % args.padded_vocab_size

                output = model.eval_batch(iter_out_of_one({"text": token_ids}), compute_loss=False)
                output_changed = model.eval_batch(iter_out_of_one({"text": token_ids_changed}), compute_loss=False)

                # All token in past should be unchanged
                torch_assert_equal(output[:, :changed_index], output_changed[:, :changed_index])
                # All tokens in the future should have changed
                self.assertFalse(
                    torch.any(equal_vectors(output[:, changed_index:], output_changed[:, changed_index:]))
                )

    def test_prefix_lm_reset_attention_mask(self):
        """
        Test prefix invariances when `reset_attention_mask=True`:
            - Past target tokens don't depend on future target tokens.
            - Target tokens depend on input tokens.
            - Input tokens depend on all other input tokens, but never target tokens.
        """
        command_args = get_default_args(self.test_file_dir_str)

        command_args["--reset-attention-mask"] = ""
        command_args["--loss-on-targets-only"] = ""

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()
                tokenizer = get_tokenizer()

                model, _, _ = setup_model_and_optimizer(pretrain_prefix_lm.model_provider)
                model = model[0]
                model._config.train_micro_batch_size_per_gpu = args.micro_batch_size
                model.set_train_batch_size(args.micro_batch_size)
                # we preprocess batch_fn manually
                model.set_batch_fn(None)

                token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

                # eod is a special token, this also guarantees that the whole row is considered as a document.
                token_ids[token_ids == tokenizer.eod] += 1
                token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

                # process batch to have non empty prefix
                input_batch, (labels, loss_mask), prefix_indices = pretrain_prefix_lm.get_batch_pipe({"text": token_ids})

                for batch_id in range(len(prefix_indices)):
                    for id in prefix_indices[batch_id]:
                        self.assertTrue(loss_mask[batch_id, id] == 1)
                        self.assertTrue(id > 0)
                        # Make sure that the last prefix token predicts the first token.
                        self.assertTrue(loss_mask[batch_id, id -1] == 1)

                output = model.eval_batch(iter_out_of_one((input_batch, (labels, loss_mask), prefix_indices)), compute_loss=False)

                ## --------------- CHANGE A TARGET TOKEN ---------------------------
                # get a modified version of the first batch
                # guaranteed to exist as each row has at least one partial document
                changed_target_index = prefix_indices[0][0]
                token_ids_changed_target = input_batch[0].clone()
                # We increment the token id on the changed index.
                token_ids_changed_target[0, changed_target_index] = \
                    (token_ids_changed_target[0, changed_target_index] + 1) % args.padded_vocab_size
                # make sure we're not changing a token to eod as it's a special token
                token_ids_changed_target[token_ids_changed_target == tokenizer.eod] += 1
                token_ids_changed_target[token_ids_changed_target == tokenizer.eod] %= args.padded_vocab_size

                # Test change
                output_changed_target = model.eval_batch(iter_out_of_one(((token_ids_changed_target, *input_batch[1:]), (labels, loss_mask), prefix_indices)), compute_loss=False)

                # All token in past should be unchanged
                torch_assert_equal(output[0, :changed_target_index], output_changed_target[0, :changed_target_index])
                # All tokens in the future should have changed
                self.assertFalse(
                    torch.any(
                        equal_vectors(output[0, changed_target_index:], output_changed_target[0, changed_target_index:])
                    )
                )
                # Unchanged changed rows should not change either
                torch_assert_equal(output[1, :], output_changed_target[1, :])

                ## --------------- CHANGE AN INPUT TOKEN ---------------------------
                # Let's change the the last prefix token and make sure that the first token changed
                # guaranteed to be positive as we avoid pathological case previously
                last_prefix_index = prefix_indices[0][0] - 1
                token_ids_changed_input = input_batch[0].clone()
                #  We increment the token id on the changed index.
                token_ids_changed_input[0, last_prefix_index] = \
                    (token_ids_changed_input[0, last_prefix_index] + 1) % args.padded_vocab_size
                # make sure we're not changing a token to eod as it's a special token
                token_ids_changed_input[token_ids_changed_input == tokenizer.eod] += 1
                token_ids_changed_input[token_ids_changed_input == tokenizer.eod] %= args.padded_vocab_size

                output_changed_input = model.eval_batch(iter_out_of_one(((token_ids_changed_input, *input_batch[1:]), (labels, loss_mask), prefix_indices)), compute_loss=False)

                # All tokens should be changed
                self.assertFalse(
                    torch.any(
                        equal_vectors(output[0, :], output_changed_input[0, :])
                    )
                )
                # Unchanged changed rows should not change either
                torch_assert_equal(output[1, :], output_changed_input[1, :])

    def test_prefix_lm_wo_reset_attention_mask(self):
        """
        Test prefix invariances when `reset_attention_mask=False`:
            - Past target tokens don't depend on future target tokens.
            - Target tokens depend on input tokens.
            - Input tokens depend on all other input tokens, but never target tokens.
        """
        command_args = get_default_args(self.test_file_dir_str)

        command_args["--loss-on-targets-only"] = ""

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()

                model, _, _ = setup_model_and_optimizer(pretrain_prefix_lm.model_provider)
                model = model[0]
                model._config.train_micro_batch_size_per_gpu = args.micro_batch_size
                model.set_train_batch_size(args.micro_batch_size)
                # we preprocess batch_fn manually
                model.set_batch_fn(None)

                token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))
                input_batch, (labels, loss_mask), prefix_indices = pretrain_prefix_lm.get_batch_pipe({"text": token_ids})

                for batch_id in range(len(prefix_indices)):
                    id = prefix_indices[batch_id]
                    self.assertTrue(loss_mask[batch_id, id] == 1)
                    self.assertTrue(id > 0)
                    # Make sure that the last prefix token predicts the first token.
                    self.assertTrue(loss_mask[batch_id, id -1] == 1)

                model.eval_batch(iter_out_of_one((input_batch, (labels, loss_mask), prefix_indices)), compute_loss=False)

                #TODO: Check all invariants

    def test_gpt_rotary_embeddings(self):
        """Test rotary embeddings"""
        command_args = get_default_args(self.test_file_dir_str)

        del command_args["--max-position-embeddings"]
        command_args["--position-embedding-type"] = "rotary"

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()
                tokenizer = get_tokenizer()

                model, _, _ = setup_model_and_optimizer(pretrain_gpt.model_provider)
                model = model[0]
                model._config.train_micro_batch_size_per_gpu = args.micro_batch_size
                model.set_train_batch_size(args.micro_batch_size)

                token_ids = torch.randint(args.padded_vocab_size, (args.micro_batch_size, args.seq_length))

                # eod is a special token
                token_ids[token_ids == tokenizer.eod] += 1
                token_ids[token_ids == tokenizer.eod] %= args.padded_vocab_size

                model.eval_batch(iter_out_of_one({"text": token_ids}), compute_loss=False)

                #TODO: Check all invariants

    @require_torch_bf16
    def test_fused_layer_norm(self):
        command_args = get_default_args(self.test_file_dir_str)

        # Condition to use custom cuda kernel
        command_args["--bf16"] = ""
        del command_args["--fp16"]

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                initialize_megatron()
                args = get_args()

                dummy_input = torch.randn(args.micro_batch_size, args.seq_length, args.hidden_size, device="cuda", dtype=torch.bfloat16)

                normalized_shape = (args.hidden_size,)
                epsilon = 1e-5
                mfln = MixedFusedLayerNorm(normalized_shape, eps=epsilon)

                self.assertTrue(mfln.use_meg_ds_fused_layer_norm, "Expected model to use Megatron-DeepSpeed custom cuda kernel for LayerNorm.")
                self.assertTrue(args.bf16, "Test has to be done in half precision.")

                # We set the weight manually so we simulate state that's not the initialisation
                weight = torch.randn(args.hidden_size, device="cuda", dtype=torch.bfloat16)
                bias = torch.randn(args.hidden_size, device="cuda", dtype=torch.bfloat16)
                mfln.weight = nn.Parameter(weight)
                mfln.bias = nn.Parameter(bias)

                mfln_output = mfln(dummy_input)
                # We check that our layernorm matches pytorch 1.11 onwards
                if version.parse(torch.__version__) >= version.parse("1.11.0"):
                    torch_layer_norm_output = F.layer_norm(dummy_input, normalized_shape, weight, bias, eps=epsilon)
                else:
                    # In this case we use can check that basically it corresponds to the fp32 version
                    torch_layer_norm_output = F.layer_norm(dummy_input.float(), normalized_shape, weight.float(), bias.float(), eps=epsilon).to(torch.bfloat16)

                torch_assert_equal(mfln_output, torch_layer_norm_output)

    @parameterized.expand([(attn_mask_type,) for attn_mask_type in AttnMaskType])
    def test_fused_masked_softmax(self, attn_mask_type: AttnMaskType):
        command_args = get_default_args(self.test_file_dir_str)

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                initialize_megatron()
                args = get_args()

                dummy_input = torch.randn(
                    args.micro_batch_size,
                    args.num_attention_heads,
                    args.seq_length,
                    args.seq_length,
                    device="cuda",
                    dtype=args.params_dtype
                )
                if attn_mask_type == AttnMaskType.causal:
                    dummy_attention_mask = None
                else:
                    dummy_attention_mask = torch.randn(
                        args.micro_batch_size,
                        1, # `args.num_attention_heads` not implemented in our cuda kernel
                        args.seq_length,
                        args.seq_length,
                        device="cuda",
                        dtype=args.params_dtype
                    ) < 0
                scale = torch.rand(())

                fused_scaled_softmax = FusedScaleMaskSoftmax(
                    input_in_fp16=args.params_dtype == torch.float16,
                    input_in_bf16=args.params_dtype == torch.bfloat16,
                    attn_mask_type=attn_mask_type,
                    scaled_masked_softmax_fusion=True,
                    mask_func=attention_mask_func,
                    softmax_in_fp32=True,
                    scale=scale,
                )
                unfused_scaled_softmax = FusedScaleMaskSoftmax(
                    input_in_fp16=args.params_dtype == torch.float16,
                    input_in_bf16=args.params_dtype == torch.bfloat16,
                    attn_mask_type=attn_mask_type,
                    scaled_masked_softmax_fusion=False,
                    mask_func=attention_mask_func,
                    softmax_in_fp32=True,
                    scale=scale,
                )

                self.assertTrue(fused_scaled_softmax.is_kernel_available(dummy_attention_mask, *dummy_input.size()))
                fused_output = fused_scaled_softmax(dummy_input, dummy_attention_mask)
                self.assertFalse(unfused_scaled_softmax.is_kernel_available(dummy_attention_mask, *dummy_input.size()))
                unfused_output = unfused_scaled_softmax(dummy_input, dummy_attention_mask)

                # Test that the nonzeros are the same with the mask
                for i in range(args.num_attention_heads):
                    if dummy_attention_mask is None:
                        # Make sure it's causal, values in the lower triangle should be not zero.
                        non_zero_values = torch.tril(torch.ones_like(fused_output[:, i]))
                        torch_assert_equal(torch.nonzero(fused_output[:, i]), torch.nonzero(non_zero_values))
                    else:
                        torch_assert_equal(torch.nonzero(fused_output[:, i]), torch.nonzero(~dummy_attention_mask[:, 0]))

                # Cuda kernel produces slightly different results
                torch_assert_close(fused_output, unfused_output)


    def test_non_causal_decoder_model_with_packed_input_passed_with_attention_mask_is_not_causal_across_segments(self):
        command_args = get_default_args(self.test_file_dir_str)
        command_args["--position-embedding-type"] = "alibi"

        with patch('sys.argv', flatten_arguments(command_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()

                args = get_args()
                tokenizer = get_tokenizer()
                # Hack: `gpt2` doesn't have a padding token, so we override that value.
                tokenizer.tokenizer.pad_token_id = tokenizer.tokenizer.eos_token_id

                data = get_dummy_mtf_decoder_packed_data(
                    micro_batch_size=args.micro_batch_size,
                    seq_length=args.seq_length,
                    vocab_size=args.padded_vocab_size,
                    special_tokens_ids={tokenizer.pad}
                )
                model, _, _ = setup_model_and_optimizer(finetune_t0_non_causal_decoder.model_provider)
                model = model[0]
                model._config.train_micro_batch_size_per_gpu = args.micro_batch_size
                model.set_train_batch_size(args.micro_batch_size)

                output = model.eval_batch(iter_out_of_one(data), compute_loss=False)

                ## --------------- CHANGE A TARGET TOKEN ---------------------------
                # change the first token in the first batch to a random value
                change_batch_id = 0
                change_token_id = 0
                token_ids_changed = data["decoder_token_ids"].clone()
                # We increment the token id on the changed index.
                token_ids_changed[change_batch_id, change_token_id] = (token_ids_changed[change_batch_id, change_token_id] + 1) % args.padded_vocab_size
                while token_ids_changed[change_batch_id, change_token_id] in {tokenizer.eod, tokenizer.pad}:
                    token_ids_changed[change_batch_id, change_token_id] = (token_ids_changed[change_batch_id, change_token_id] + 1) % args.padded_vocab_size

                # Test change
                output_changed_target = model.eval_batch(iter_out_of_one({**data, "decoder_token_ids": token_ids_changed}), compute_loss=False)

                first_segment_first_batch_id_end = (torch.nonzero(data["decoder_segment_ids"][change_batch_id, 1:] - data["decoder_segment_ids"][change_batch_id, :-1]) + 1)[0]
                # Check that values changed in segment 1 of batch_id 0
                self.assertFalse(torch.any(
                    equal_vectors(
                        output[change_batch_id, change_token_id:first_segment_first_batch_id_end],
                        output_changed_target[change_batch_id, change_token_id:first_segment_first_batch_id_end]
                    )
                ))
                # Check that values did not change in other segments of batch_id 0
                torch_assert_equal(
                    output[change_batch_id, first_segment_first_batch_id_end:],
                    output_changed_target[change_batch_id, first_segment_first_batch_id_end:]
                )
                # Check that values did not change in other segments in other batches
                non_change_ids = torch.arange(output.shape[0]) != change_batch_id
                torch_assert_equal(output[non_change_ids], output_changed_target[non_change_ids])

                ## --------------- CHANGE A TARGET TOKEN ---------------------------
                # change the last token in the first batch to a pad
                token_ids_changed_pad = data["decoder_token_ids"].clone()
                segment_ids_changed_pad = data["decoder_segment_ids"].clone()
                # We increment the token id on the changed index.
                token_ids_changed_pad[change_batch_id, -1] = tokenizer.pad
                segment_ids_changed_pad[change_batch_id, -1] = 0

                # Test model handles padding correctly
                output_changed_pad = model.eval_batch(iter_out_of_one({**data, "decoder_token_ids": token_ids_changed_pad, "decoder_segment_ids": segment_ids_changed_pad}), compute_loss=False)

                self.assertFalse(torch.any(torch.isnan(output_changed_pad)))
