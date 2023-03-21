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

import filecmp
import io
import json
import re
import os
import unittest
import functools

from pathlib import Path

from megatron.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    set_seed
)

from datasets import load_dataset

set_seed(42)


def write_jsonl(path, lines_num=1000, line_length=1024):
    def get_text_line(line_length):
        # XXX: fix to generate line_length
        return "It's a wonderful world. I'm just walking on air. Talk of heaven on earth. I've got more than my share. Haven't got a care. Happy all day through. It's a wonderful world. Loving wonderful you!"

    with io.open(path, "w", encoding="utf-8") as f:

        for i in range(lines_num):
            rec = dict(text=get_text_line(line_length))
            x = json.dumps(rec, indent=0, ensure_ascii=False)
            x = re.sub(r'\n', ' ', x, 0, re.M)
            f.write(x + "\n")

@functools.lru_cache()
def download_hf_dataset(dsetname):
    return load_dataset(dsetname)

class MegDSTestPreprocessing(TestCasePlus):
    """ """

    def setUp(self):
        super().setUp()


    def test_preprocess_data(self):
        src_dir = self.src_dir
        data_dir = f"{self.data_dir}/gpt2"
        output_dir = self.get_auto_remove_tmp_dir() # "./xxx", after=False)

        # autogenerate "input.jsonl"
        input_path = f"{output_dir}/input.jsonl"
        write_jsonl(input_path)

        output_prefix =f"{output_dir}/test-ds"

        cmd = f"""
        python {src_dir}/tools/preprocess_data.py
            --input {input_path}
            --output-prefix {output_prefix}
            --dataset-impl mmap
            --tokenizer-type GPT2BPETokenizer
            --merge-file {data_dir}/gpt2-tiny-merges.txt
            --vocab {data_dir}/gpt2-tiny-vocab.json
            --append-eod
            --workers 2
        """.split()

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        for ext in ["bin", "idx"]:
            tgt_path = f"{output_prefix}_text_document.{ext}"
            self.assertTrue(Path(tgt_path).exists(), )

    def compare_meg_data_files(self, tgt, ref):
        for ext in ["bin", "idx"]:
            tgt_path = f"{tgt}.{ext}"
            ref_path = f"{ref}.{ext}"
            self.assertTrue(Path(tgt_path).exists(), )
            self.assertTrue(filecmp.cmp(tgt_path, ref_path, shallow=False))

    def preprocess_partitioned_dataset(self, output_dir, dsetname, splitname, linelimit, numparts):
        """Preprocess a dataset as a whole and in shards to prepare environment for merge test.

        Load specified HF dataset using given split and record limit.
        Write the dataset to a jsonl file and preprocess.
        Also split dataset into numparts contiguous shards, write each shard to its own jsonl, and preprocess each.
        Return path to the full dataset and a list of paths for each shard."""

        src_dir = self.src_dir
        data_dir = f"{self.data_dir}/gpt2"

        # preproces_data_dist requires one to have already downloaded the input HF dataset.
        # We do that by running this script before the test.
        dset = download_hf_dataset(dsetname)[splitname]

        # limit the test to use the first linelimit entries to be faster
        dset = dset.select(range(linelimit))

        # write jsonl file of full dataset
        json_ds = f"{output_dir}/ds-full.jsonl"
        dset.to_json(json_ds)

        # process full jsonl into indexed dataset file
        ds_full = f"{output_dir}/ds-full"
        cmd = f"""
                python {src_dir}/tools/preprocess_data.py
                    --input {json_ds}
                    --output-prefix {ds_full}
                    --dataset-impl mmap
                    --tokenizer-type GPT2BPETokenizer
                    --merge-file {data_dir}/gpt2-tiny-merges.txt
                    --vocab {data_dir}/gpt2-tiny-vocab.json
                    --append-eod
                """.split()
        ds_full += '_text_document'

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        # write each part to its own json file
        ds_parts = []
        for i in range(numparts):
            json_part = f"{output_dir}/ds-part-{i}.jsonl"
            dset.shard(numparts, i, contiguous=True).to_json(json_part)

            ds_part = f"{output_dir}/ds-part-{i}"
            ds_parts.append(ds_part + '_text_document')
            cmd = f"""
                    python {src_dir}/tools/preprocess_data.py
                        --input {json_part}
                        --output-prefix {ds_part}
                        --dataset-impl mmap
                        --tokenizer-type GPT2BPETokenizer
                        --merge-file {data_dir}/gpt2-tiny-merges.txt
                        --vocab {data_dir}/gpt2-tiny-vocab.json
                        --append-eod
                    """.split()

            # keep for quick debug
            # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
            execute_subprocess_async(cmd, env=self.get_env())

        return ds_full, ds_parts

    def test_merge_serial(self):
        """Check that serial merge of partial dataset files produces the same file as the full dataset."""
        src_dir = self.src_dir
        output_dir = self.get_auto_remove_tmp_dir()  # "./xxx", after=False)

        # process full dataset, and process the full dataset as 3 contiguous chunks
        ds_full, ds_parts = self.preprocess_partitioned_dataset(output_dir, 'stas/openwebtext-10k', 'train', 100, 3)

        # merge the part files into a single indexed dataset
        ds_merged = f"{output_dir}/ds-merged"
        cmd = f"""
                python {src_dir}/tools/merge_preprocessed_data.py
                    --datasets {" ".join(ds_parts)}
                    --output-prefix {ds_merged}
                """.split()

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        # the full dataset and the merged dataset should be identical
        self.compare_meg_data_files(ds_full, ds_merged)

    def test_merge_distributed(self):
        """Check that serial merge of partial dataset files produces the same file as the full dataset."""
        src_dir = self.src_dir
        output_dir = self.get_auto_remove_tmp_dir()  # "./xxx", after=False)

        # process full dataset, and process the full dataset as 3 contiguous chunks
        ds_full, ds_parts = self.preprocess_partitioned_dataset(output_dir, 'stas/openwebtext-10k', 'train', 100, 3)

        # merge the part files into a single indexed dataset
        ds_merged = f"{output_dir}/ds-merged"
        cmd = f"""
                python -m torch.distributed.launch --nproc_per_node 6 {src_dir}/tools/merge_preprocessed_data.py
                    --merge distributed
                    --datasets {" ".join(ds_parts)}
                    --output-prefix {ds_merged}
                    --torch-backend gloo
                """.split()

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        # the full dataset and the merged dataset should be identical
        self.compare_meg_data_files(ds_full, ds_merged)

    def test_process_data_microsoft(self):
        """We want to be stable to Microsoft version."""
        src_dir = self.src_dir
        data_dir = f"{self.data_dir}/gpt2"
        output_dir = self.get_auto_remove_tmp_dir()  # "./xxx", after=False)

        input_path = f"{self.tests_dir}/data/gpt2/openwebtext-1000.jsonl"

        output_prefix = f"{output_dir}/test-ds-meg-gpt2-openwebtext"

        cmd = f"""
                python {src_dir}/tools/preprocess_data.py
                    --input {input_path}
                    --output-prefix {output_prefix}
                    --dataset-impl mmap
                    --tokenizer-type GPT2BPETokenizer
                    --merge-file {data_dir}/gpt2-tiny-merges.txt
                    --vocab {data_dir}/gpt2-tiny-vocab.json
                    --append-eod
                    --workers 2
                """.split()

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        self.compare_meg_data_files(f"{output_prefix}_text_document", f"{data_dir}/meg-gpt2-openwebtext_text_document")

    def test_process_data_dist_microsoft(self):
        """We want to be stable to Microsoft version."""
        src_dir = self.src_dir
        data_dir = f"{self.data_dir}/gpt2"
        output_dir = self.get_auto_remove_tmp_dir()  # "./xxx", after=False)

        output_prefix = f"{output_dir}/test-ds-meg-gpt2-openwebtext_1k"

        # preprocess_data_dist requires one to have already downloaded the input HF dataset.
        # We do that by running this script before the test.
        dsetname = 'stas/openwebtext-10k'
        download_hf_dataset(dsetname)

        cmd = f"""
                python -m torch.distributed.launch --nproc_per_node 2 {src_dir}/tools/preprocess_data_dist.py
                    --input {dsetname}
                    --count 1000
                    --output-prefix {output_prefix}
                    --dataset-impl mmap
                    --tokenizer-type GPT2BPETokenizer
                    --merge-file {data_dir}/gpt2-tiny-merges.txt
                    --vocab {data_dir}/gpt2-tiny-vocab.json
                    --append-eod
                """.split()

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        self.compare_meg_data_files(f"{output_prefix}_text_document", f"{data_dir}/meg-gpt2-openwebtext_text_document")

    def test_process_data_dist_serial_microsoft(self):
        """We want to be stable to Microsoft version."""
        src_dir = self.src_dir
        data_dir = f"{self.data_dir}/gpt2"
        output_dir = self.get_auto_remove_tmp_dir()  # "./xxx", after=False)

        output_prefix = f"{output_dir}/test-ds-meg-gpt2-openwebtext_1k"

        # preproces_data_dist requires one to have already downloaded the input HF dataset.
        # We do that by running this script before the test.
        dsetname = 'stas/openwebtext-10k'
        download_hf_dataset(dsetname)

        cmd = f"""
                python -m torch.distributed.launch --nproc_per_node 2 {src_dir}/tools/preprocess_data_dist.py
                    --input {dsetname}
                    --count 1000
                    --merge serial
                    --output-prefix {output_prefix}
                    --dataset-impl mmap
                    --tokenizer-type GPT2BPETokenizer
                    --merge-file {data_dir}/gpt2-tiny-merges.txt
                    --vocab {data_dir}/gpt2-tiny-vocab.json
                    --append-eod
                """.split()

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        self.compare_meg_data_files(f"{output_prefix}_text_document", f"{data_dir}/meg-gpt2-openwebtext_text_document")
