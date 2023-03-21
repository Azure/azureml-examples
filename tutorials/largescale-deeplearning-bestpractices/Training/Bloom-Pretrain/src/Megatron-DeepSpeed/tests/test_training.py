# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import dataclasses
import io
import json
import os
import glob
import re
import shutil
import unittest
from pathlib import Path
from parameterized import parameterized

from megatron.testing_utils import (
    CaptureStdout,
    CaptureStd,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    require_bnb_non_decorator,
    require_deepspeed,
    require_torch_gpu,
    set_seed
)

set_seed(42)


def get_launcher(num_gpus):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    return f"deepspeed --num_nodes 1 --num_gpus {num_gpus}".split()

def get_3d_dimensions():
    num_gpus = get_gpu_count()

    # with fewer gpus the preference is first to to do PP>1, then TP>1, then DP>1
    if num_gpus >= 8:
        dp_size = 2
        pp_size = 2
        tp_size = 2
    if num_gpus >= 4:
        dp_size = 1
        pp_size = 2
        tp_size = 2
    elif num_gpus >= 2:
        dp_size = 1
        pp_size = 2
        tp_size = 1
    else:
        dp_size = 1
        pp_size = 1
        tp_size = 1

    return pp_size, tp_size, dp_size


@require_deepspeed
@require_torch_gpu
class MegDSTestTraining(TestCasePlus):
    """ """

    def setUp(self):
        super().setUp()

        # at times magatron fails to build kernels and doesn't remove the lock file, which makes
        # subsequent runs hang - so make sure there is no lock when starting the testing
        meg_lock_file_path = self.repo_root_dir_str + "/megatron/fused_kernels/build/lock"
        if os.path.exists(meg_lock_file_path):
            os.unlink(meg_lock_file_path)

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

    def get_variation_config(self, variation, output_dir, n_samples=None):
        data_dir = self.copy_data_to_temp(self.data_dir,"gpt2")

        pp_size, tp_size, dp_size = get_3d_dimensions()
        num_gpus = pp_size * tp_size * dp_size
        print(f"Using {num_gpus} GPUs")

        if variation == "bnb":
            # we want to make sure at least tp=2 is used, so we swap tp and pp
            pp_size, tp_size = tp_size, pp_size

        if n_samples is None:
            n_samples = 300 # about 56 iterations

        exit_interval = 20 # some samples in the first half and then some more in the 2nd half after resume
        seq_len = 128

        # common/shared configs

        ds_args = f"""
                --deepspeed
                --deepspeed_config {self.test_file_dir_str}/ds_config.json
                --zero-stage 1
                --deepspeed-activation-checkpointing
        """.split()

        args = f"""
                --tensor-model-parallel-size {tp_size}
                --pipeline-model-parallel-size {pp_size}
                --distributed-backend nccl

                --log-interval 1
                --save-interval 10
                --eval-interval 10
                --eval-iters 5
                --checkpoint-activations
                --partition-activations
                --exit-interval {exit_interval}

                --merge-file {data_dir}/gpt2-tiny-merges.txt
                --vocab-file {data_dir}/gpt2-tiny-vocab.json
                --save {output_dir}/checkpoints
                --load {output_dir}/checkpoints
                --data-path {data_dir}/meg-gpt2-openwebtext_text_document
                --tensorboard-dir {output_dir}/tensorboard
                --tensorboard-queue-size 5
                --log-timers-to-tensorboard
                --log-batch-size-to-tensorboard
                --log-validation-ppl-to-tensorboard

                --num-layers 2
                --hidden-size 64
                --num-attention-heads 2
                --seq-length {seq_len}
                --max-position-embeddings 1024
                --micro-batch-size 1
                --global-batch-size 16

                --optimizer adam
                --adam-beta1 0.9
                --adam-beta2 0.95
                --adam-eps 1e-8
                --lr 1e-4
                --lr-warmup-samples 5
                --clip-grad 1.0
                --weight-decay 1e-1
                --embed-layernorm
                --sync-tp-duplicated-parameters
                --fp16

                --log-level debug
                --log-level-replica info
        """.split()


        if variation == "base":

            new_args = f"""
                --rampup-batch-size 2 2 {n_samples}
                --train-samples {n_samples}

                --lr-decay-samples 6

            """.split()

            new_ds_args = f"""
                --deepspeed_config {self.test_file_dir_str}/ds_config.json
            """.split()


        elif variation == "bnb":
            # BitsAndBytes - 8-bit optimizer

            new_args = f"""
                --rampup-batch-size 2 2 {n_samples}
                --train-samples {n_samples}

                --lr-decay-samples 6

                --use-bnb-optimizer
            """.split()

            new_ds_args = f"""
                --deepspeed_config {self.test_file_dir_str}/ds_config.json
            """.split()


        elif variation == "cl":
            # CurriculumLearning

            lr_decay_samples = 6
            lr_decay_tokens = lr_decay_samples * seq_len

            train_tokens = n_samples * seq_len

            # XXX: if changing seq_len from 128, must adjust ds config to:
            #  curriculum_learning.max_difficulty: $SEQLEN

            # XXX: probably we should write the ds config on the fly to keep everything in sync,
            # rather than using the pre-saved config

            new_args = f"""
                --train-samples {n_samples*2}
                --train-tokens {train_tokens}

                --lr-decay-tokens {lr_decay_tokens}
            """.split()

            new_ds_args = f"""
                --deepspeed_config {self.test_file_dir_str}/ds_config_cl.json
            """.split()

        elif variation == "glu":
            new_args = f"""
                --rampup-batch-size 2 2 {n_samples}
                --train-samples {n_samples}

                --lr-decay-samples 6

                --no-bias-gelu-fusion
                --glu-activation geglu
            """.split()

            new_ds_args = f"""
                --deepspeed_config {self.test_file_dir_str}/ds_config.json
            """.split()

        elif variation == "alibi":
            new_args = f"""
                --rampup-batch-size 2 2 {n_samples}
                --train-samples {n_samples}

                --lr-decay-samples 6

                --position-embedding-type alibi
            """.split()

            new_ds_args = f"""
                --deepspeed_config {self.test_file_dir_str}/ds_config.json
            """.split()

        else:
            raise ValueError(f"Don't know of variation {variation}")

        args.extend(new_args)
        ds_args.extend(new_ds_args)

        return args, ds_args, num_gpus

    def test_kill_switch(self):

        variation = "base"

        src_dir = self.src_dir
        output_dir = self.get_auto_remove_tmp_dir() # "./xxx", after=False)
        kill_switch_path = os.path.join(output_dir, "kill-switch-xyz")
        args, ds_args, num_gpus = self.get_variation_config(variation, output_dir)
        args += f"--kill-switch-path {kill_switch_path}".split()

        script = [f"{src_dir}/pretrain_gpt.py"]
        launcher = get_launcher(num_gpus)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        # 1. kill switch armed but not triggered
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test deepspeed is running
        self.assertIn("DeepSpeed info", cs.out)

        # 2. trigger kill switch
        fh = open(kill_switch_path, "w")
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        self.assertIn(f"Detected kill switch at {kill_switch_path}", cs.out)

        # test deepspeed wasn't run
        self.assertNotIn("DeepSpeed info", cs.out)


    @parameterized.expand(["base", "cl", "bnb", "glu", "alibi"])
    def test_training_all(self, variation):

        # optional runs
        if variation == "bnb":
            require_bnb_non_decorator()

        # all in one test
        src_dir = self.src_dir
        output_dir = self.get_auto_remove_tmp_dir() # "./xxx", after=False)

        args, ds_args, num_gpus = self.get_variation_config(variation, output_dir)

        script = [f"{src_dir}/pretrain_gpt.py"]
        launcher = get_launcher(num_gpus)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        # 1. test training from scratch (no checkpoint)
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test deepspeed is running
        self.assertIn("DeepSpeed info", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test there should be no checkpoint this round
        self.assertIn(f"Unable to find latest file at {output_dir}/checkpoints/latest", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

        # test tensorboard
        tensorboard_files = glob.glob(f"{output_dir}/tensorboard/events*")
        self.assertEqual(len(tensorboard_files), 1, "tensorboard files")

        if variation == "glu":
            self.assertIn("Using GLU activation: GELU", cs.out)

        if variation == "alibi":
            self.assertIn("Using Alibi", cs.out)

        # 2. test training from checkpoint: resume
        # now do it again, this time resuming from the checkpoint
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test checkpoint loading
        self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

        # test tensorboard (1 file from the first run, plus 1 now)
        tensorboard_files = glob.glob(f"{output_dir}/tensorboard/events*")
        self.assertEqual(len(tensorboard_files), 2, "tensorboard files")

        if variation == "glu":
            self.assertIn("Using GLU activation: GELU", cs.out)

    @parameterized.expand([(True, True), (False, False), (True, False), (False, True)])
    def test_training_prefix_lm_all(self, loss_on_targets_only, reweight_loss_based_on_position_frequency):
        # all in one test
        src_dir = self.src_dir
        data_dir = self.copy_data_to_temp(self.data_dir,"gpt2")

        output_dir = self.get_auto_remove_tmp_dir() # "./xxx", after=False)
        logs_dir = f"{output_dir}/logs"
        Path(logs_dir).mkdir(parents=True, exist_ok=True)

        pp_size, tp_size, dp_size = get_3d_dimensions()
        num_gpus = pp_size * tp_size * dp_size

        n_samples = 200 # about 37 iterations
        exit_interval = 20 # some samples in the first half and then some more in the 2nd half after resume
        args = f"""
            --tensor-model-parallel-size {tp_size}
            --pipeline-model-parallel-size {pp_size}
            --distributed-backend nccl

            --num-layers 2
            --hidden-size 64
            --num-attention-heads 2
            --seq-length 128
            --max-position-embeddings 1024
            --micro-batch-size 1
            --rampup-batch-size 2 2 {n_samples}
            --global-batch-size 16
            --train-samples {n_samples}
            {"--loss-on-targets-only" if loss_on_targets_only else ""}
            {"--reweight-loss-based-on-position-frequency" if reweight_loss_based_on_position_frequency else ""}

            --optimizer adam
            --adam-beta1 0.9
            --adam-beta2 0.95
            --adam-eps 1e-8
            --lr 1e-4
            --lr-warmup-samples 5
            --clip-grad 1.0
            --weight-decay 1e-1
            --fp16

            --log-interval 5
            --save-interval 10
            --eval-interval 10
            --eval-iters 5
            --checkpoint-activations
            --exit-interval {exit_interval}

            --merge-file {data_dir}/gpt2-tiny-merges.txt
            --vocab-file {data_dir}/gpt2-tiny-vocab.json
            --log-path {logs_dir}
            --save {output_dir}/checkpoints
            --load {output_dir}/checkpoints
            --data-path {data_dir}/meg-gpt2-openwebtext_text_document
            --tensorboard-dir {output_dir}/tensorboard
            --tensorboard-queue-size 5
            --log-timers-to-tensorboard
            --log-batch-size-to-tensorboard
            --log-validation-ppl-to-tensorboard

            --log-level debug
        """.split()

        ds_args = f"""
            --deepspeed
            --deepspeed_config {self.test_file_dir_str}/ds_config.json
            --zero-stage 1
            --deepspeed-activation-checkpointing
        """.split()

        script = [f"{src_dir}/pretrain_prefix_lm.py"]
        launcher = get_launcher(num_gpus)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        # 1. test training from scratch (no checkpoint)
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test deepspeed is running
        self.assertIn("DeepSpeed info", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test there should be no checkpoint this round
        self.assertIn(f"Unable to find latest file at {output_dir}/checkpoints/latest", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

        # test tensorboard
        tensorboard_files = glob.glob(f"{output_dir}/tensorboard/events*")
        self.assertEqual(len(tensorboard_files), 1, "tensorboard files")

        if reweight_loss_based_on_position_frequency:
            self.assertIn("Using loss reweighting", cs.out)

        # 2. test training from checkpoint: resume
        # now do it again, this time resuming from the checkpoint
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test checkpoint loading
        self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

        # test tensorboard (1 file from the first run, plus 1 now)
        tensorboard_files = glob.glob(f"{output_dir}/tensorboard/events*")
        self.assertEqual(len(tensorboard_files), 2, "tensorboard files")

    def test_training_t0(self):
        data_path = self.copy_data_to_temp(self.data_dir, "gpt2/ag_news_prompt")
        output_dir = self.get_auto_remove_tmp_dir()
        logs_dir = f"{output_dir}/logs"
        Path(logs_dir).mkdir(parents=True, exist_ok=True)

        pp_size, tp_size, dp_size = get_3d_dimensions()
        num_gpus = pp_size * tp_size * dp_size

        n_samples = 200 # about 37 iterations
        exit_interval = 10 # some samples in the first half and then some more in the 2nd half after resume

        args = f"""
            --tensor-model-parallel-size {tp_size}
            --pipeline-model-parallel-size {pp_size}
            --distributed-backend nccl

            --num-layers 2
            --hidden-size 64
            --num-attention-heads 2
            --seq-length 128
            --max-position-embeddings 1024
            --position-embedding-type alibi
            --micro-batch-size 1
            --rampup-batch-size 2 2 {n_samples}
            --global-batch-size 16
            --train-samples {n_samples}

            --optimizer adam
            --adam-beta1 0.9
            --adam-beta2 0.95
            --adam-eps 1e-8
            --lr 1e-4
            --lr-warmup-samples 5
            --clip-grad 1.0
            --weight-decay 1e-1
            --fp16

            --log-interval 5
            --save-interval 10
            --eval-interval 10
            --eval-iters 5
            --checkpoint-activations
            --exit-interval {exit_interval}
            --tokenizer-type PretrainedFromHF
            --tokenizer-name-or-path bigscience/tokenizer
            --log-path {logs_dir}
            --save {output_dir}/checkpoints
            --load {output_dir}/checkpoints
            --data-path {data_path}
            --split 90,10,0
            --tensorboard-dir {output_dir}/tensorboard
            --tensorboard-queue-size 5
            --log-timers-to-tensorboard
            --log-batch-size-to-tensorboard
            --log-validation-ppl-to-tensorboard

            --log-level debug
        """.split()

        ds_args = f"""
            --deepspeed
            --deepspeed_config {self.test_file_dir_str}/ds_config.json
            --zero-stage 1
            --deepspeed-activation-checkpointing
        """.split()

        script = [f"{self.src_dir}/finetune_t0_non_causal_decoder.py"]
        launcher = get_launcher(num_gpus)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        # 1. test training from scratch (no checkpoint)
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test deepspeed is running
        self.assertIn("DeepSpeed info", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test there should be no checkpoint this round
        self.assertIn(f"Unable to find latest file at {output_dir}/checkpoints/latest", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

        # test tensorboard
        tensorboard_files = glob.glob(f"{output_dir}/tensorboard/events*")
        self.assertEqual(len(tensorboard_files), 1, "tensorboard files")

        # 2. test training from checkpoint: resume
        # now do it again, this time resuming from the checkpoint
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test checkpoint loading
        self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

        # test tensorboard (1 file from the first run, plus 1 now)
        tensorboard_files = glob.glob(f"{output_dir}/tensorboard/events*")
        self.assertEqual(len(tensorboard_files), 2, "tensorboard files")

    @parameterized.expand(["gpt", "prefix", "no_eval"])
    def test_mode2_dataloading(self, variation):
        src_dir = self.src_dir
        data_dir = self.copy_data_to_temp(self.data_dir, "gpt2")
        output_dir = self.get_auto_remove_tmp_dir() # "./xxx", after=False)
        logs_dir = f"{output_dir}/logs"
        Path(logs_dir).mkdir(parents=True, exist_ok=True)

        pp_size, tp_size, dp_size = get_3d_dimensions()
        num_gpus = pp_size * tp_size * dp_size

        n_samples = 200 # about 37 iterations
        exit_interval = 20 # some samples in the first half and then some more in the 2nd half after resume
        args = f"""
            --tensor-model-parallel-size {tp_size}
            --pipeline-model-parallel-size {pp_size}
            --distributed-backend nccl

            --num-layers 2
            --hidden-size 64
            --num-attention-heads 2
            --seq-length 128
            --max-position-embeddings 1024
            --micro-batch-size 1
            --rampup-batch-size 2 2 {n_samples}
            --global-batch-size 16
            --train-samples {n_samples}
            --loss-on-targets-only

            --optimizer adam
            --adam-beta1 0.9
            --adam-beta2 0.95
            --adam-eps 1e-8
            --lr 1e-4
            --lr-warmup-samples 5
            --clip-grad 1.0
            --weight-decay 1e-1
            --fp16

            --log-interval 5
            --save-interval 10
            --eval-interval 10
            --eval-iters 5
            --checkpoint-activations
            --exit-interval {exit_interval}

            --merge-file {data_dir}/gpt2-tiny-merges.txt
            --vocab-file {data_dir}/gpt2-tiny-vocab.json
            --log-path {logs_dir}
            --save {output_dir}/checkpoints
            --tensorboard-dir {output_dir}/tensorboard
            --tensorboard-queue-size 5
            --log-timers-to-tensorboard
            --log-batch-size-to-tensorboard
            --log-validation-ppl-to-tensorboard
        """.split()

        data_args = [
            "--train-weighted-split-paths", f'TRAIN: 1 0:0.95 {data_dir}/meg-gpt2-openwebtext_text_document, 0.3 0:0.90 {data_dir}/meg-gpt2-openwebtext_text_document']

        if variation != "no_eval":
            data_args += ["--valid-weighted-split-paths", f'VALID1: 1 0.95:0.98 {data_dir}/meg-gpt2-openwebtext_text_document, 0.3 0.90:0.99 {data_dir}/meg-gpt2-openwebtext_text_document',
                                            f'VALID2: 0.5 0.95:0.97 {data_dir}/meg-gpt2-openwebtext_text_document, 0.5 0.90:0.98 {data_dir}/meg-gpt2-openwebtext_text_document']

        ds_args = f"""
            --deepspeed
            --deepspeed_config {self.test_file_dir_str}/ds_config.json
            --zero-stage 1
            --deepspeed-activation-checkpointing
        """.split()

        if variation == "prefix":
            script = [f"{src_dir}/pretrain_prefix_lm.py"]
        else:
            script = [f"{src_dir}/pretrain_gpt.py"]
        launcher = get_launcher(num_gpus)

        cmd = launcher + script + args + data_args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        # 1. test training from scratch (no checkpoint)
        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test deepspeed is running
        self.assertIn("DeepSpeed info", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

        # test tensorboard
        tensorboard_files = glob.glob(f"{output_dir}/tensorboard/events*")
        self.assertEqual(len(tensorboard_files), 1, "tensorboard files")

    def test_skip_train_iteration(self):
        # skip iterations setup
        extra_args = f"""
            --skip-train-iteration-range 2-2 4-7
        """.split()

        src_dir = self.src_dir
        output_dir = self.get_auto_remove_tmp_dir()
        args, ds_args, num_gpus = self.get_variation_config("base", output_dir, n_samples=200)
        args.extend(extra_args)
        script = [f"{src_dir}/pretrain_gpt.py"]
        launcher = get_launcher(num_gpus)
        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # check skipped iterations
        self.assertIn("Skipped iterations 2 to 2 due to --skip-train-iteration-range flag", cs.out)
        self.assertIn("Skipped iterations 4 to 7 due to --skip-train-iteration-range flag", cs.out)

        train_iterations = range(1,10)
        for i in train_iterations:
            self.assertTrue(f"iteration {i:8d}/" in cs.out)
