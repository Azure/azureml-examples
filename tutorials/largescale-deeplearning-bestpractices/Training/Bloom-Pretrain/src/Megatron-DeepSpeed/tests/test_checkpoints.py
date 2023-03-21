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

import io
import os
import pytest
from pathlib import Path

from parameterized import parameterized
from megatron.testing_utils import (
    CaptureStdout,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    require_deepspeed,
    require_torch_gpu,
    require_torch_multi_gpu,
    set_seed
)

set_seed(42)


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_to_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"

params = [
    # TP_PP_DP
    ["1_1_1", "1_1_1"],
    ["2_1_1", "1_1_1"],
    ["1_2_1", "1_1_1"],
    ["1_1_2", "1_1_1"],

    ["2_1_1", "2_1_1"],
    ["1_1_1", "2_1_1"],
    ["1_1_1", "1_2_1"],
    ["1_1_1", "1_1_2"],

    ["1_1_2", "1_1_2"],
    ["1_1_2", "2_1_1"],
    ["1_1_2", "1_2_1"],

    ["1_2_1", "1_2_1"],
    ["1_2_1", "2_1_1"],
    ["1_2_1", "1_1_2"],

    ["2_1_1", "2_1_1"],
    ["2_1_1", "1_2_1"],
    ["2_1_1", "1_1_2"],

    ["2_2_2", "1_1_1"],
    ["2_2_2", "2_2_2"],
    ["1_1_1", "2_2_2"],

    ["1_1_8", "2_2_2"],

]

def get_launcher(num_gpus):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    return f"deepspeed --num_nodes 1 --num_gpus {num_gpus}".split()

@require_deepspeed
@require_torch_gpu
class MegDSTestCheckpoints(TestCasePlus):
    """ """

    def setUp(self):
        super().setUp()

        # at times magatron fails to build kernels and doesn't remove the lock file, which makes
        # subsequent runs hang - so make sure there is no lock when starting the testing
        meg_lock_file_path = self.repo_root_dir_str + "/megatron/fused_kernels/build/lock"
        if os.path.exists(meg_lock_file_path):
            os.unlink(meg_lock_file_path)

    def get_config(self, output_dir, tp_size, pp_size, dp_size):
        data_dir = f"{self.data_dir}/gpt2"

        num_gpus = pp_size * tp_size * dp_size
        print(f"Using {num_gpus} GPUs")

        n_samples = 300 # about 56 iterations

        exit_interval = 20 # some samples in the first half and then some more in the 2nd half after resume
        seq_len = 128

        # XXX: for now while testing shapes make it really short and fast
        exit_interval = 1
        seq_len = 8


        # common/shared configs

        ds_args = f"""
                --deepspeed
                --deepspeed_config {self.test_file_dir_str}/ds_config_bf16.json
                --zero-stage 0
                --deepspeed-activation-checkpointing
        """.split()

        args = f"""
                --tensor-model-parallel-size {tp_size}
                --pipeline-model-parallel-size {pp_size}
                --distributed-backend nccl

                --log-interval 1
                --save-interval 1
                --eval-interval 10
                --eval-iters 1
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
                --hidden-size 8
                --num-attention-heads 2
                --seq-length {seq_len}
                --max-position-embeddings 8
                --micro-batch-size 1
                --global-batch-size 16
                --train-samples {n_samples}

                --embed-layernorm
                --position-embedding-type alibi

                --optimizer adam
                --adam-beta1 0.9
                --adam-beta2 0.95
                --adam-eps 1e-8
                --lr 1e-4
                --lr-warmup-samples 5
                --lr-decay-samples 6
                --clip-grad 1.0
                --weight-decay 1e-1
                --bf16

                --log-level debug
                --log-level-replica info
        """.split()


        # XXX: fails to handle:
        #--embed-layernorm
        #
# stderr: RuntimeError: Error(s) in loading state_dict for VocabParallelEmbedding:
# stderr:         size mismatch for norm.weight: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).
# stderr:         size mismatch for norm.bias: copying a param with shape torch.Size([128]) from checkpoint, the shape in current model is torch.Size([64]).

        return args, ds_args, num_gpus


    def train_checkpoint(self, output_dir, tp_size=1, pp_size=1, dp_size=1):
        src_dir = self.src_dir
        script = [f"{src_dir}/pretrain_gpt.py"]

        args, ds_args, num_gpus = self.get_config(output_dir, tp_size, pp_size, dp_size)
        launcher = get_launcher(num_gpus)
        cmd = launcher + script + args + ds_args
        # keep for quick debug
        #print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

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

    def convert_checkpoint_to_universal(self, output_dir, step):
        cmd = f"""
            python tools/convert_checkpoint/ds_to_universal.py
            --input_folder  {output_dir}/checkpoints/global_step{step}
            --output_folder {output_dir}/checkpoints/global_step{step}_universal
        """.split()
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        self.assertIn("Convert DeepSpeed Checkpoint to Universal Checkpoint", cs.out)

    def resume_from_checkpoint(self, output_dir, tp_size=1, pp_size=1, dp_size=1):
        src_dir = self.src_dir
        script = [f"{src_dir}/pretrain_gpt.py"]

        args, ds_args, num_gpus = self.get_config(output_dir, tp_size, pp_size, dp_size)
        launcher = get_launcher(num_gpus)
        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test checkpoint loading
        self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)

    def resume_from_universal_checkpoint(self, output_dir, tp_size=1, pp_size=1, dp_size=1):
        src_dir = self.src_dir
        script = [f"{src_dir}/pretrain_gpt.py"]

        args, ds_args, num_gpus = self.get_config(output_dir, tp_size, pp_size, dp_size)
        launcher = get_launcher(num_gpus)
        cmd = launcher + script + args + ds_args + ["--universal-checkpoint"]
        # keep for quick debug
        #print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die

        with CaptureStdout() as cs:
            execute_subprocess_async(cmd, env=self.get_env())

        # test checkpoint loading
        self.assertIn(f"successfully loaded checkpoint from {output_dir}/checkpoints", cs.out)

        # test reports
        self.assertIn("consumed samples", cs.out)

        # test checkpoint saving
        self.assertIn("successfully saved checkpoint at iteration", cs.out)


    @require_torch_multi_gpu
    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_checkpoint_reshaping_main(self, src, tgt):
        # this test needs at least 2 gpus - if there are more gpus it will do more extensive testing

        tp_size_src, pp_size_src, dp_size_src = list(map(int, src.split('_')))
        tp_size_tgt, pp_size_tgt, dp_size_tgt = list(map(int, tgt.split('_')))

        n_gpus = get_gpu_count()
        n_gpus_src = tp_size_src * pp_size_src * dp_size_src
        n_gpus_tgt = tp_size_tgt * pp_size_tgt * dp_size_tgt

        if n_gpus_src > n_gpus:
            pytest.skip(f"the test requires {n_gpus_src} gpus for source topology but have only {n_gpus}")
        if n_gpus_tgt > n_gpus:
            pytest.skip(f"the test requires {n_gpus_tgt} gpus for target topology but have only {n_gpus}")

        output_dir = self.get_auto_remove_tmp_dir("./xxx", after=False)

        # 1. train with initial topology defined in the first arg of params
        self.train_checkpoint(output_dir, tp_size=tp_size_src , pp_size=pp_size_src , dp_size=dp_size_src )

        # 2. convert checkpoint to universal checkpoint (topology )
        self.convert_checkpoint_to_universal(output_dir=output_dir, step=1)

        # 3. check we can resume training from a reshaped checkpoint to the target topology - the last arg of params
        self.resume_from_universal_checkpoint(output_dir, tp_size=tp_size_tgt, pp_size=pp_size_tgt, dp_size=dp_size_tgt)


    @require_torch_multi_gpu
    def test_checkpoint_reshaping_empty_dir(self):

        output_dir = self.get_auto_remove_tmp_dir() # "./xxx", after=False)
        with self.assertRaises(RuntimeError) as context:
            self.convert_checkpoint_to_universal(output_dir=output_dir, step=1)
