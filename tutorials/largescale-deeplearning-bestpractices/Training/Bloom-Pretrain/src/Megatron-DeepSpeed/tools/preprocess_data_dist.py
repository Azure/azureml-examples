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

"""Processing data for pretraining.

This builds data files from a source HuggingFace dataset, e.g,

  from datasets import load_dataset
  dset = load_dataset('openwebtext', split='train')

The implementation uses `torch.distributed` for inter-process communication,
and it assumes that files are written to a global file system, such that one process
can read a file written by another process.

A list of sample index values from the source dataset are selected
by rank 0 and scattered to all ranks.
Each process tokenizes a subset of samples and writes its output to a part file.
After all ranks have finished, the part files are merged into a final output file.

One may optionally use storage local to each process to store the part file.
For example, on a Linux cluster, one might write the part file to /dev/shm.

To run:

python -m torch.distributed.launch --nproc_per_node 40 --nnodes 8 \
    preprocess_data_dist.py \
        --input openwebtext \
        --output-prefix openwebtext-bert \
        --vocab bert-large-uncased-vocab.txt \
        --dataset-impl mmap \
        --tokenizer-type BertWordPieceLowerCase \
        --split-sentences
"""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import stat
import time

import numpy as np
import random

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from datasets import config, logging, load_dataset
from datasets.utils.file_utils import OfflineModeIsEnabled

from megatron.tokenizer import build_tokenizer
from megatron.data.indexed_dataset import data_file_path, index_file_path, make_builder, best_fitting_dtype, gather_files_dist
from megatron.data.distdata import DistData

def msg(msg, flush=False):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{timestamp}: {msg}", flush=flush)

def msgerr(msg, flush=False):
    print(f"ERROR: {msg}", flush=flush)

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

        self.tokenizer = build_tokenizer(self.args)

        if self.args.split_sentences:
            if not nltk_available:
                msgerr("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                self.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                self.splitter = splitter
        else:
            self.splitter = IdentitySplitter()

    def encode_text(self, text):
        ids = {}
        for key in self.args.columns:
            doc_ids = []
            for sentence in self.splitter.tokenize(text):
                sentence_ids = self.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0:
                if self.args.append_eod:
                    doc_ids[-1].append(self.tokenizer.eod)
                ids[key] = doc_ids
        return ids, len(text)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Dataset name')
    group.add_argument('--split', type=str, default='train',
                       help='Dataset split to select.')
    group.add_argument('--columns', nargs='+', default=['text'],
                       help='Space separate listed of column names to extract from dataset')
    group.add_argument('--count', type=int, default=None,
                       help='Limit the number of samples to select.')
    group.add_argument('--shuffle', action='store_true',
                       help='Shuffle samples before writing output files.')
    group.add_argument('--seed', type=int, default=None,
                       help='Seed to pass to random.seed for shuffle operations.')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None, 
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to be divisible by this value.'
                            'Value of the size of the vocabulary of the tokenizer to reach. This value must be greater than'
                            ' the initial size of the tokenizer. If this argument is used the value of '
                            '`make-vocab-size-divisible-by` will be ignored.')


    group = parser.add_argument_group(title='runtime')
    group.add_argument('--torch-backend', type=str, default='gloo', choices=['gloo', 'mpi'],
                       help='Select torch.distributed backend.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='Local rank of calling process on its node (from torch.distributed.launch).')
    group.add_argument('--merge', type=str, default='parallel', choices=['parallel', 'serial', 'both'],
                       help=('Method to merge intermediate per-rank files into the final data files.  '
                             'With "parallel", each rank writes directly to the final files, '
                             'while rank 0 copies data from all per-rank files with "serial".  '
                             'A parallel merge can be faster, but for correctness, it requires the underlying file system '
                             'to support parallel write operations to a file that is shared among multiple processes.  '
                             'One can choose "both" for testing purposes, in which case the final files written '
                             'by the parallel method are given an additional ".par" extension.'))
    group.add_argument('--scratch', type=str, default=None,
                       help=('Path to local storage on compute nodes to write per-rank files before merging, like /dev/shm.  '
                             'One can only use this option with a parallel merge.'))
    group.add_argument('--log-interval', type=int, default=30,
                       help='Seconds between progress updates (0 to disable)')

    args = parser.parse_args()
    args.keep_empty = False

    # initialize our distributed environment
    args.distctx = DistData(backend=args.torch_backend)

    # some functions like build_tokenizer use args.rank to filter stdout messages
    args.rank = args.distctx.rank
    args.numranks = args.distctx.numranks

    # some default/dummy values for the tokenizer
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            if args.rank == 0:
                msg("Bert tokenizer detected, are you sure you don't want to split sentences?")

    args.level = "document"
    if args.split_sentences:
        args.level = "sentence"

    # TODO: perhaps more user friendly to disable scratch and print a warning?
    # check that serial merge is not attempted with scratch
    if args.scratch is not None and args.merge != 'parallel':
        raise  ValueError("The --scratch option is only valid with --merge=parallel")

    return args

def format_byterate(byterate):
    mbps = byterate / (1024.0 * 1024.0)
    return f"{mbps:0.3f} MB/s"

def load_dset(args):
    # Avoid downloading datasets unless explicitly requested.
    # We allow the user to override this behavior if they set $HF_DATASETS_OFFLINE.
    if 'HF_DATASETS_OFFLINE' not in os.environ:
        # To disable downloads, we could set $HF_DATASETS_OFFLINE=1.
        # However, the HF_DATASETS_OFFLINE environment variable is processed
        # when the datasets module is imported, so it must be set before the import statement.
        # sets HF_DATASETS_OFFLINE within the environment of this script
        #os.environ['HF_DATASETS_OFFLINE'] = "1"

        # Alternatively, one can set datasets.config.HF_DATASETS_OFFLINE=1.
        # That seems to work even after the import statement,
        # though this usage is not documented.
        config.HF_DATASETS_OFFLINE = 1

    # silence info messages from all procs except rank 0 
    if args.rank != 0:
        logging.set_verbosity(logging.ERROR)

    time_start = time.time()

    # Load the specified HuggingFace dataset.
    # Give rank 0 a head start in case the dataset is not already cached.
    err = None
    dsetname = args.input
    if args.rank == 0:
        msg(f"Opening dataset {dsetname}")
        try:
            dset = load_dataset(dsetname, split=args.split, keep_in_memory=None)
        except OfflineModeIsEnabled as e:
            msgerr(f"Cannot download '{dsetname}' since running in offline mode.")
            msgerr(f"If the dataset is large, it may be more efficient to download with a single process:")
            msgerr(f"    from datasets import load_dataset")
            msgerr(f"    dset = load_dataset('{dsetname}')")
            msgerr(f"Alternatively, one can force this script to download by setting $HF_DATASETS_OFFLINE=0", flush=True)
            err = e
        except Exception as e:
            msgerr(f"Unexpected error: {sys.exc_info()[0]}", flush=True)
            err = e

    # determine whether rank 0 succeeded in loading the dataset
    args.distctx.allraise_if(err)

    # Rank 0 succeeded, attempt to load dataset on all other ranks.
    # This should load from cache now.
    if args.rank != 0:
        try:
            dset = load_dataset(dsetname, split=args.split, keep_in_memory=None)
        except Exception as e:
            # this print might be noisy, but better than nothing
            msgerr(f"Unexpected error: {sys.exc_info()[0]}", flush=True)
            err = e

    # verify that all ranks loaded the dataset
    args.distctx.allraise_if(err)

    time_end = time.time()
    if args.rank == 0:
        msg(f"Seconds to load dataset: {time_end - time_start}", flush=True)

    return dset

def get_num_samples(args, dset_size):
    """Given a dataset size and optional count argument, return number of samples to process."""
    num_samples = dset_size
    if args.count is not None and args.count < dset_size:
        num_samples = args.count
    return num_samples

def select_sample_list(args, dset_size):
    """Given the total number of samples, select a list of sample index values"""
    # determine total number of samples that we'll read
    num_samples = get_num_samples(args, dset_size)

    # create sample index list on rank 0,
    # optionally shuffle the list,
    # and optionally limit the sample count
    time_select = time.time()
    idxlist = None
    if args.rank == 0:
        # generate a list of all index values
        idxlist = np.arange(dset_size, dtype=np.int64)

        # optionally shuffle
        if args.shuffle:
            # args.seed may be an int (to seed) or None (to not)
            rng = np.random.default_rng(args.seed)
            rng.shuffle(idxlist)

        # optionally limit the sample count
        if args.count is not None:
            idxlist = idxlist[:num_samples]

    # get a list of the number of elements each rank will hold
    counts = get_proc_counts(num_samples, args.numranks)

    # scatter sample index values from rank 0 to all procs
    # based on distribution defined in counts list
    time_bcast = time.time()
    idx = args.distctx.scatterv_(idxlist, counts, root=0)

    args.distctx.barrier()
    time_end = time.time()
    if args.rank == 0:
        msg(f"Select index stats:")
        msg(f"    Shuffle: {args.shuffle}")
        msg(f"    Seconds to select: {time_bcast - time_select}")
        msg(f"    Seconds to broadcast: {time_end - time_bcast}")
        msg(f"    Seconds total: {time_end - time_select}", flush=True)

    return idx

def get_proc_counts(num, num_ranks):
    num_per_rank, remainder = divmod(num, num_ranks)
    return [num_per_rank + 1 if rank < remainder else num_per_rank for rank in range(num_ranks)]

def get_filename(args, key, rank=None):
    pathname = args.output_prefix

    # redirect per-rank file to scratch dir if defined
    if args.scratch is not None and rank is not None:
        basename = os.path.basename(pathname)
        pathname = os.path.join(args.scratch, basename)

    if rank is not None:
        filename = f"{pathname}_{key}_{args.level}_{rank}"
    else:
        filename = f"{pathname}_{key}_{args.level}"

    return filename

def rank_files_write(args, dset, idx, encoder):
    time_start = time.time()

    # compute total number of samples we'e processing
    num_samples = get_num_samples(args, len(dset))

    # we'll total up the number of docs, sentences, and bytes
    # processed across all ranks
    dset_stats = np.zeros(3, dtype=np.int64) # docs, sentences, bytes

    # we'll set this to false on any problem
    err = None
    times = np.zeros(3, dtype=np.float32) # read, tokenize, write
    try:
        # create data file for each rank
        if args.rank == 0:
            msg(f"Vocab size: {args.padded_vocab_size}")
            msg(f"Output prefix: {args.output_prefix}")
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in args.columns:
            filebase = get_filename(args, key, args.rank)
            output_bin_files[key] = data_file_path(filebase)
            output_idx_files[key] = index_file_path(filebase)
            best_dtype = best_fitting_dtype(args.padded_vocab_size) if args.dataset_impl == "mmap" else None
            builders[key] = make_builder(output_bin_files[key],
                                         impl=args.dataset_impl,
                                         dtype=best_dtype)

        # each rank tokenizes its samples and writes its own file
        progress_next = time.time() + float(args.log_interval)
        for i in idx:
            sample_id = int(i)
            for key in args.columns:
                # tokenize text for the given sample index
                start_read = time.time()
                text = dset[sample_id][key]
                start_encode = time.time()
                doc, bytes_processed = encoder.encode_text(text)

                # add tokenized sequence to our data file
                start_write = time.time()
                for key, sentences in doc.items():
                    for sentence in sentences:
                        builders[key].add_item(torch.IntTensor(sentence))
                    builders[key].end_document()
                    dset_stats[0] += 1
                    dset_stats[1] += len(sentences)
                dset_stats[2] += bytes_processed
                end_write = time.time()

                times[0] += start_encode - start_read
                times[1] += start_write - start_encode
                times[2] += end_write - start_write

            if args.rank == 0 and args.log_interval > 0 and time.time() > progress_next:
                current = time.time()
                progress_next = current + float(args.log_interval)

                elapsed = current - time_start
                docs = dset_stats[0] * args.numranks
                percent = docs / num_samples * 100.0
                docrate = docs / elapsed if elapsed > 0.0 else 0.0
                mbs = dset_stats[2] * args.numranks / elapsed / 1024 / 1024 if elapsed > 0.0 else 0.0
                secs_left = int((num_samples - docs) / docrate if docrate > 0.0 else 0.0)
                msg(f"Processed (estimated) {docs} of {num_samples} docs ({percent:0.2f}%) in {int(elapsed)} secs, "
                    f"{docrate:0.3f} docs/s, {mbs:0.3f} MB/s, "
                    f"{secs_left} secs left ...",
                    flush=True)

        # finalize file of each rank
        for key in args.columns:
            builders[key].finalize(output_idx_files[key])
            del builders[key] # file closed in __del__
    except Exception as e:
        # caught an exception, assume our file is invalid
        err = e

    # In case rank 0 finishes early and stops printing progress messages,
    # inform user that it's waiting for other ranks to finish.
    if args.rank == 0 and args.log_interval > 0:
        msg(f"Waiting for ranks to finalize files ...", flush=True)

    # wait for all ranks to finish their files
    args.distctx.barrier()
    time_end = time.time()

    # compute total stats across all processes
    args.distctx.all_sum_(times)
    args.distctx.all_sum_(dset_stats)
    if args.rank == 0:
        secs = time_end - time_start
        docrate = dset_stats[0] / secs if secs > 0.0 else 0.0
        sentrate = dset_stats[1] / secs if secs > 0.0 else 0.0
        byterate = dset_stats[2] / secs if secs > 0.0 else 0.0
        secs_read_per_sample = times[0] / dset_stats[0] if dset_stats[0] > 0 else 0.0
        secs_encode_per_sample = times[1] / dset_stats[0] if dset_stats[0] > 0 else 0.0
        secs_write_per_sample = times[2] / dset_stats[0] if dset_stats[0] > 0 else 0.0
        msg("Process stats:")
        msg(f"    Seconds to process: {secs}")
        msg(f"    {dset_stats[0]} docs {docrate} docs/sec")
        msg(f"    {dset_stats[1]} sents {sentrate} sents/sec")
        msg(f"    {dset_stats[2]} bytes {format_byterate(byterate)}")
        msg(f"    Total read seconds {times[0]}, {secs_read_per_sample} sec/sample")
        msg(f"    Total encode seconds {times[1]}, {secs_encode_per_sample} sec/sample")
        msg(f"    Total write seconds {times[2]}, {secs_write_per_sample} sec/sample")

    # check whether all ranks wrote their part successfully
    args.distctx.allraise_if(err)

def rank_files_merge_parallel(args):
    """Each process directly writes its portion of the data from its per-rank file into the final file."""
    merge_start = time.time()
    numbytes = np.zeros(1, dtype=np.int64)
    for key in args.columns:
        # merge the per-rank file from each process into a single shared file
        filemain = get_filename(args, key)
        filerank = get_filename(args, key, args.rank)
        gather_files_dist(filemain, [filerank], args.distctx)

        # total up bytes read during the merge
        binfilerank = data_file_path(filerank)
        idxfilerank = index_file_path(filerank)
        numbytes[0] += os.stat(binfilerank)[stat.ST_SIZE]
        numbytes[0] += os.stat(idxfilerank)[stat.ST_SIZE]

        # If user want to use both a parallel and serial merge (for testing),
        # rename the parallel output files so that the serial merge does not clobber them.
        if args.merge == 'both' and args.rank == 0:
            binfilemain = data_file_path(filemain)
            idxfilemain = index_file_path(filemain)
            os.rename(binfilemain, binfilemain + ".par")
            os.rename(idxfilemain, idxfilemain + ".par")

    # Total up number of bytes read across all ranks,
    # and wait on all ranks before stopping the timer.
    args.distctx.all_sum_(numbytes)
    merge_end = time.time()
    if args.rank == 0:
        secs = merge_end - merge_start
        byterate = numbytes[0] / secs if secs > 0.0 else 0.0
        msg("Parallel merge stats:")
        msg(f"    Scratch: {args.scratch}")
        msg(f"    Seconds to merge: {secs}")
        msg(f"    {int(numbytes)} bytes {format_byterate(byterate)}")

def rank_files_merge_serial(args):
    """Rank 0 merges data from all per-rank files into the final file."""
    if args.rank == 0:
        msg("Merging rank files ...", flush=True)
        merge_start = time.time()
        numbytes = 0

        # define name of single file
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in args.columns:
            filebase = get_filename(args, key)
            output_bin_files[key] = data_file_path(filebase)
            output_idx_files[key] = index_file_path(filebase)
            best_dtype = best_fitting_dtype(args.padded_vocab_size) if args.dataset_impl == "mmap" else None
            builders[key] = make_builder(output_bin_files[key],
                                         impl=args.dataset_impl,
                                         dtype=best_dtype)

        # merge all ranks into one file
        for rank in range(args.numranks):
            for key in args.columns:
                infile = get_filename(args, key, rank)
                builders[key].merge_file_(infile)

                # sum up the number of merged bytes
                binfile = data_file_path(infile)
                idxfile = index_file_path(infile)
                numbytes += os.stat(binfile)[stat.ST_SIZE]
                numbytes += os.stat(idxfile)[stat.ST_SIZE]

        # finalize the merged file
        msg("Finalizing merged file ...", flush=True)
        for key in args.columns:
            builders[key].finalize(output_idx_files[key])
            del builders[key] # file closed in __del__

        merge_end = time.time()
        secs = merge_end - merge_start
        byterate = numbytes / secs if secs > 0.0 else 0.0
        msg(f"Merged {args.numranks} files into {args.output_prefix}")
        msg("Serial merge stats:")
        msg(f"    Seconds to merge: {secs}")
        msg(f"    {numbytes} bytes {format_byterate(byterate)}")

    # hold everyone until rank 0 is done
    args.distctx.barrier()

def rank_files_merge(args):
    # use parallel merge if asked
    if args.merge in ['parallel', 'both']:
        rank_files_merge_parallel(args)

    # if using node-local storage, skip sequential merge
    if args.scratch is not None:
        return

    # can fall back to a serial merge
    if args.merge in ['serial', 'both']:
        rank_files_merge_serial(args)

def rank_files_delete(args):
    # delete per-rank files
    if args.rank == 0:
        msg("Deleting rank files ...", flush=True)

    for key in args.columns:
        filebase = get_filename(args, key, args.rank)

        binfile = data_file_path(filebase)
        if os.path.exists(binfile):
            os.remove(binfile)

        idxfile = index_file_path(filebase)
        if os.path.exists(idxfile):
            os.remove(idxfile)

    # hold everyone until all are done
    args.distctx.barrier()

def main():
    args = get_args()
    startup_start = time.time()

    # load the dataset
    dset = load_dset(args)
    if args.rank == 0:
        print(dset)
        msg(f"Processing features: {args.columns}")

    # create sample index list,
    # optionally shuffle the list,
    # and optionally limit the sample count
    idx = select_sample_list(args, len(dset))

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)

    # wait for all ranks before stopping timer
    args.distctx.barrier()
    startup_end = time.time()
    if args.rank == 0:
        msg(f"Seconds to startup: {startup_end - startup_start}")

    # have each rank write its file,
    # all ranks should raise an exception if any rank has a problem
    try:
        rank_files_write(args, dset, idx, encoder)
    except Exception as e:
        # If any process fails, we skip the merge since the resulting file would be invalid.
        # We still delete files to clean up, since those might be invalid anyway.
        if args.rank == 0:
            msgerr(f"At least one process failed to write its file, skipping merge and cleaning up", flush=True)

        # delete per-rank files, do this even on error
        rank_files_delete(args)

        # re-raise exception caught during write phase
        raise e

    # all ranks were successful writing their file, merge them into one
    rank_files_merge(args)

    # delete per-rank files
    rank_files_delete(args)

    end_time = time.time()
    if args.rank == 0:
        msg(f"Runtime: {end_time - startup_start} secs", flush=True)
        msg(f"Done")

if __name__ == '__main__':
    main()
