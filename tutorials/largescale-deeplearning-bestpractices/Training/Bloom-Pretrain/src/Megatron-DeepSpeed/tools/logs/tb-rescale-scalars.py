#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script rescales scalar values in TensorBoard log files.
# It does the modification in-place (so make back ups!).
#
# Example:
#
# find . -name "*.tfevents*" -exec tb-rescale-scalars.py {} "iteration-time/samples per second" 1000 \;
#
# More than one old tag can be rescaled – use ";" as a separator:
#
# tb-rescale-scalars.py events.out.tfevents.1 "training loss;validation loss" 1e-2
#
# By default, BigScience GPT throughput values will be fixed up according to
# https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/236,
# i.e. the rescaling fixes values wrongly logged as "seconds" when they are
# actually milliseconds.
#
# This script is derived from https://stackoverflow.com/a/60080531/9201239
# and https://gist.github.com/stas00/4cd1651d1c8f01196ea322c733bde46c.

import os
import sys

# Use this if you want to avoid using the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event


def rescale_scalars(input_file, tags, rescale_factor):
    new_file = input_file + '.new'
    # Make a record writer
    with tf.io.TFRecordWriter(new_file) as writer:
        # Iterate event records
        for rec in tf.data.TFRecordDataset([input_file]):
            # Read event
            ev = Event()
            ev.MergeFromString(rec.numpy())
            # Check if it is a summary
            if ev.summary:
                # Iterate summary values
                for v in ev.summary.value:
                    # Check if the tag should be rescaled
                    if v.tag in tags:
                        v.simple_value *= rescale_factor
            writer.write(ev.SerializeToString())
    os.rename(new_file, input_file)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'{sys.argv[0]} <input file> [<tags> [<rescale factor>]]',
              file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) < 3:
        sys.argv.append(';'.join([
            'iteration-time/samples per second',
            'iteration-time/samples per second per replica',
            'iteration-time/tokens per second',
            'iteration-time/tokens per second per replica',
        ]))
    if len(sys.argv) < 4:
        sys.argv.append('1000')

    input_file, tags, rescale_factor = sys.argv[1:]
    tags = tags.split(';')
    rescale_factor = float(rescale_factor)
    rescale_scalars(input_file, tags, rescale_factor)
    print('Done')
