#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script renames event names in TensorBoard log files.
# It does the renaming in-place (so make back ups!).
#
# Example:
#
# find . -name "*.tfevents*" -exec tb-rename-events.py {} "iteration-time" "iteration-time/iteration-time" \;
#
# More than one old tag can be remapped to one new tag – use ";" as a separator:
#
# tb-rename-events.py events.out.tfevents.1 "training loss;validation loss" "loss"
#
# This script is derived from https://stackoverflow.com/a/60080531/9201239
# and https://gist.github.com/stas00/4cd1651d1c8f01196ea322c733bde46c.

import os
import sys

# Use this if you want to avoid using the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event


def rename_events(input_file, old_tags, new_tag):
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
                    # Check if the tag should be renamed
                    if v.tag in old_tags:
                        # Rename with new tag name
                        v.tag = new_tag
            writer.write(ev.SerializeToString())
    os.rename(new_file, input_file)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'{sys.argv[0]} <input file> <old tags> <new tag>',
              file=sys.stderr)
        sys.exit(1)
    input_file, old_tags, new_tag = sys.argv[1:]
    old_tags = old_tags.split(';')
    rename_events(input_file, old_tags, new_tag)
    print('Done')
