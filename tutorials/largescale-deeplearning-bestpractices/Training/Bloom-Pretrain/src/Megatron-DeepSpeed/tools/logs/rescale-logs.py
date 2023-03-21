#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script fixes up BigScience log files by adjusting and fixing
# units of logged values to be seconds instead of milliseconds.
# It does the modification in-place (so make back ups!).
#
# Example:
#
# find . -name "*.out*" -print0 | xargs -0 -P 8 rescale-logs.py
#
# See also the discussion in
# https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/236.
#
# This script is derived from https://stackoverflow.com/a/60080531/9201239
# and https://gist.github.com/stas00/4cd1651d1c8f01196ea322c733bde46c.

import os
import re
import sys

LINE_START_RE = re.compile(' ?iteration')
ELAPSED_TIME_RE = re.compile(r'elapsed time per iteration \(ms\): ([0-9.]+)')
SAMPLES_PER_SEC_RE = re.compile('samples per second: ([0-9.]+)')


def rescale_logs(log_file_path):
    new_log_file_path = log_file_path + '.new'
    with open(log_file_path, 'r') as log_file:
        with open(new_log_file_path, 'w') as new_log_file:
            for line in log_file.readlines():
                if LINE_START_RE.match(line):
                    match = ELAPSED_TIME_RE.search(line)
                    if match:
                        # Logged time is in ms, so convert the match.
                        time_in_sec = float(match[1]) / 1000
                        replacement = (
                            f'elapsed time per iteration (s): '
                            f'{time_in_sec:.2f}'
                        )

                        # We only need to replace once per line.
                        line = ELAPSED_TIME_RE.sub(replacement, line, count=1)

                    match = SAMPLES_PER_SEC_RE.search(line)
                    if match:
                        # Logged time is in ms, so convert the match.
                        time_in_sec = float(match[1]) * 1000
                        # As the values are already logged up to 3
                        # numbers after the decimal point and we scale
                        # by exactly that amount, we log them without
                        # decimal point here in order to not seem more
                        # exact than we are.
                        replacement = f'samples per second: {time_in_sec:.0f}'

                        # We only need to replace once per line.
                        line = SAMPLES_PER_SEC_RE.sub(
                            replacement,
                            line,
                            count=1,
                        )

                new_log_file.write(line)

    os.rename(new_log_file_path, log_file_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'{sys.argv[0]} <input file>',
              file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    rescale_logs(input_file)
    print('Done')
