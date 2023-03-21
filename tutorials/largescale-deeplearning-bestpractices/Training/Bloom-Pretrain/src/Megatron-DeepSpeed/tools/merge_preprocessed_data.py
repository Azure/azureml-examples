"""Merge a list of indexed datasets into a single indexed dataset.

This script can run in two modes: a serial mode in which a single
process merges all datasets, and a distributed parallel mode in
which a set of processes in a torch.distributed environment
collectively merge datasets into a single file.

The serial mode is simpler to use.

Provides that the file system permits it, the parallel mode
can improve performance when merging many dataset files.
The distributed mode requires one to write the output dataset to
a POSIX-complaint file system that supports shared parallel
access to the file as different processes write to different
regions of the output file simultaneously.

To run in serial mode:

  python tools/merge_preprocessed_data.py \
    --datasets \
      meg-gpt2-oscar-en-500-p1_text_document \
      meg-gpt2-oscar-en-500-p2_text_document \
      meg-gpt2-oscar-en-500-p3_text_document \
    --output-prefix meg-gpt2_oscar_text_document

To run in distributed mode:

  MASTER_ADDR="localhost"
  MASTER_PORT=12345
  python -m torch.distributed.launch \
      --nproc_per_node 40 \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
    tools/merge_preprocessed_data.py \
      --merge distributed \
      --datasets \
        meg-gpt2-oscar-en-500-p1_text_document \
        meg-gpt2-oscar-en-500-p2_text_document \
        meg-gpt2-oscar-en-500-p3_text_document \
      --output-prefix meg-gpt2_oscar_text_document
"""

import argparse
import time

from megatron import print_rank_0
from megatron.data import indexed_dataset
from megatron.data.indexed_dataset import infer_dataset_impl, MMapIndexedDataset, data_file_path, index_file_path, merge_files_dist
from megatron.data.distdata import DistData


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--merge', type=str, default='serial', choices=['serial', 'distributed'],
                       help='Whether to use a serial merge with a single process or a distributed parallel merge.')
    group.add_argument('--torch-backend', type=str, default=None, choices=['gloo', 'mpi'],
                       help='Select torch.distributed backend.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='Local rank of calling process on its node (from torch.distributed.launch).')

    args = parser.parse_args()

    # initialize distributed environment if distributed merge requested
    if args.merge == 'distributed':
        if args.torch_backend is None:
            print_rank_0("Distributed merge using --torch-backend=gloo as default")
            args.torch_backend = 'gloo'
        args.distctx = DistData(backend=args.torch_backend)

    if args.merge == 'serial' and args.torch_backend is not None:
        print_rank_0("Ignoring setting for --torch-backend since using a serial merge")

    return args

def main():
    """
    Allows merging multiple types of datasets generated through preprocess_data script
    """
    args = get_args()
    startup_start = time.time()

    print_rank_0(f"Merging {args.datasets}")
    print_rank_0(f"Output prefix: {args.output_prefix}")

    if args.merge == 'distributed':
        if args.distctx.numranks > len(args.datasets):
            print_rank_0(f"Using more ranks {args.distctx.numranks} than datasets {len(args.datasets)}")
        merge_files_dist(args.output_prefix, args.datasets, args.distctx)
    else:
        # We use the first dataset to infer the dataset implementation common to all datasets.
        dataset_impl = infer_dataset_impl(args.datasets[0])
        assert dataset_impl is not None

        # Ensure that all datasets use the same implementaton.
        for ds in args.datasets:
            ds_impl = infer_dataset_impl(ds)
            assert ds_impl == dataset_impl, f"Dataset type '{ds_impl}' in file '{ds}' does not match type '{dataset_impl}' from file '{args.datasets[0]}'"

        # We use the first dataset to infer the dtype common to all datasets.
        first_dataset = indexed_dataset.make_dataset(args.datasets[0], dataset_impl)
        dtype = first_dataset.dtype if isinstance(first_dataset, MMapIndexedDataset) else None

        output_filename = args.output_prefix
        output_bin_file = data_file_path(output_filename)
        output_idx_file = index_file_path(output_filename)
        builder = indexed_dataset.make_builder(output_bin_file,
                                               impl=dataset_impl,
                                               dtype=dtype)
        for dataset in args.datasets:
            builder.merge_file_(dataset)

        builder.finalize(output_idx_file)

    startup_end = time.time()
    print_rank_0(f"Time to merge: {startup_end - startup_start}")
    print_rank_0(f"Merged {len(args.datasets)} datasets to {args.output_prefix}")

if __name__ == "__main__":
    main()
