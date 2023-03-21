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

"""Megatron global variables."""

import functools
import os
import sys
import time

from packaging import version
from pathlib import Path

import torch

from megatron.tokenizer import build_tokenizer
from .arguments import parse_args
from .microbatches import build_num_microbatches_calculator

_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_CODECARBON_TRACKER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None

def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,
                                               consistency_check)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER

def get_codecarbon_tracker():
    """Return codecarbon tracker. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_CODECARBON_TRACKER

def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS


def set_global_variables(extra_args_provider=None, args_defaults={},
                         ignore_unknown_args=False):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""
    args = _parse_args(extra_args_provider=extra_args_provider,
                       defaults=args_defaults,
                       ignore_unknown_args=ignore_unknown_args)
    _build_num_microbatches_calculator(args)
    if args.vocab_file or args.tokenizer_name_or_path:
        _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_codecarbon_tracker(args)
    _set_adlr_autoresume(args)
    _set_timers()


def _parse_args(extra_args_provider=None, defaults={},
                ignore_unknown_args=False):
    """Parse entire arguments."""
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _GLOBAL_ARGS = parse_args(extra_args_provider=extra_args_provider,
                              defaults=defaults,
                              ignore_unknown_args=ignore_unknown_args)
    return _GLOBAL_ARGS


def _build_num_microbatches_calculator(args):

    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR,
                                   'num microbatches calculator')

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        args)


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir and args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
            # this is supposed to make the data load in TB faster
            if version.parse(torch.__version__) >= version.parse("1.9"):
                _GLOBAL_TENSORBOARD_WRITER.add_scalar = functools.partial(
                    _GLOBAL_TENSORBOARD_WRITER.add_scalar, new_style=True
                )
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


# Important: the codecarbon is very unstable and its latest incarnation using the python scheduler interferes with the asyncio library we use in the test suite which breaks everything, so making this a no-op for now.
def _set_codecarbon_tracker(args):

    return # turned off

    global _GLOBAL_CODECARBON_TRACKER
    if not hasattr(args, 'codecarbon_dir') or args.codecarbon_dir is None:
        return

    import codecarbon
    if args.rank == 0:
        print('> setting codecarbon ...')

    output_dir = args.codecarbon_dir
    output_file = f"emissions-{args.rank:03d}.csv"
    logger_preamble = f"r{args.rank:03d}"
    log_level = "error"
    country_iso_code = "FRA"

    # CC was emitting all kinds of warnings about issues with measurements, so the following
    # settings are supposed to help
    misfire_grace_time = 3
    measure_power_secs = 60
    max_instances = 3

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _GLOBAL_CODECARBON_TRACKER = codecarbon.OfflineEmissionsTracker(
        output_dir=output_dir,
        output_file=output_file,
        logger_preamble=logger_preamble,
        log_level=log_level,
        misfire_grace_time=misfire_grace_time,
        measure_power_secs=measure_power_secs,
        max_instances=max_instances,
        country_iso_code=country_iso_code,
    )


def codecarbon_tracker_start():

    return # turned off, see the notes above

    global _GLOBAL_CODECARBON_TRACKER
    if _GLOBAL_CODECARBON_TRACKER is None:
        return

    #print("CC START")
    _GLOBAL_CODECARBON_TRACKER.start()


def codecarbon_tracker_stop():

    return # turned off, see the notes above

    global _GLOBAL_CODECARBON_TRACKER
    if _GLOBAL_CODECARBON_TRACKER is None:
        return

    #print("CC STOP")
    _GLOBAL_CODECARBON_TRACKER.stop()


def codecarbon_tracker_flush():

    return # turned off, see the notes above

    global _GLOBAL_CODECARBON_TRACKER
    if _GLOBAL_CODECARBON_TRACKER is None:
        return

    #print("CC FLUSH")
    _GLOBAL_CODECARBON_TRACKER.flush()


def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    if args.adlr_autoresume:
        if args.rank == 0:
            print('enabling autoresume ...', flush=True)
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except BaseException:
            print('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(f'time/{name}-time', value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = ''
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if not len(string):
            return
        string = 'time (ms)' + string
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                    torch.distributed.get_world_size() - 1):
                print(string, flush=True)
        else:
            print(string, flush=True)
