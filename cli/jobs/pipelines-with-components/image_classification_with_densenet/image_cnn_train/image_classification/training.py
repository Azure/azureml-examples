# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import logger as log
from . import resnet as models
from . import utils
import dllogger

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DIRECTORY_TO_WATCH = "/usr/share"
checkpoint_file_name = "checkpoint_backup.pth.tar"

from multiprocessing import Value
from ctypes import c_bool

import mlflow

mlflow.autolog()


class PreemptHandler(FileSystemEventHandler):
    def __init__(self):
        super(PreemptHandler, self).__init__()
        self.is_preempted = Value(c_bool, False)

    def on_any_event(self, event):
        if not event.is_directory and event.src_path.endswith("/to-be-preempted"):
            print(datetime.utcnow(), "Detected Preempt Signal, should stop and return.")
            self.is_preempted.value = True


class PreemptDetector:
    def __init__(self):
        self.observer = Observer()
        self.event_handler = PreemptHandler()

    def run(self):
        self.observer.schedule(self.event_handler, DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()

    def is_preempted(self):
        return self.event_handler.is_preempted.value == True

    def stop(self):
        self.observer.stop()


ACC_METADATA = {"unit": "%", "format": ":.2f"}
IPS_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}
LOSS_METADATA = {"format": ":.5f"}


class ModelAndLoss(nn.Module):
    def __init__(self, arch, loss, pretrained_weights=None, cuda=True, fp16=False):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        print("=> creating model '{}'".format(arch))
        model = models.build_resnet(arch[0], arch[1])
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        if cuda:
            model = model.cuda()
        if fp16:
            model = network_to_half(model)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    def distributed(self):
        self.model = DDP(self.model)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)


def get_optimizer(
    parameters,
    fp16,
    lr,
    momentum,
    weight_decay,
    nesterov=False,
    state=None,
    static_loss_scale=1.0,
    dynamic_loss_scale=False,
    bn_weight_decay=False,
):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        optimizer = torch.optim.SGD(
            [v for n, v in parameters],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if "bn" in n]
        rest_params = [v for n, v in parameters if not "bn" in n]
        print(len(bn_params))
        print(len(rest_params))
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": 0},
                {"params": rest_params, "weight_decay": weight_decay},
            ],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if fp16:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=static_loss_scale,
            dynamic_loss_scale=dynamic_loss_scale,
            verbose=False,
        )

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric(
            "lr", log.LR_METER(), verbosity=dllogger.Verbosity.VERBOSE
        )

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric("lr", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(
    base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None
):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier) / es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay**e)
        return lr

    return lr_policy(_lr_fn, logger=logger)


def get_train_step(
    model_and_loss, optimizer, fp16, use_amp=False, batch_size_multiplier=1
):
    def _step(input, target, optimizer_step=True):
        input_var = Variable(input)
        target_var = Variable(target)
        loss, output = model_and_loss(input_var, target_var)
        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if optimizer_step:
            opt = (
                optimizer.optimizer
                if isinstance(optimizer, FP16_Optimizer)
                else optimizer
            )
            for param_group in opt.param_groups:
                for param in param_group["params"]:
                    param.grad /= batch_size_multiplier

            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def train(
    train_loader,
    model_and_loss,
    optimizer,
    lr_scheduler,
    fp16,
    logger,
    epoch,
    detector,
    use_amp=False,
    prof=-1,
    batch_size_multiplier=1,
    register_metrics=True,
    total_train_step=0,
    writer=None,
):
    print(f"training...")
    print(f"register_metrics {register_metrics}, logger {logger}.")
    if register_metrics and logger is not None:
        logger.register_metric(
            "train.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "train.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "train.compute_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_train_step(
        model_and_loss,
        optimizer,
        fp16,
        use_amp=use_amp,
        batch_size_multiplier=batch_size_multiplier,
    )

    model_and_loss.train()
    end = time.time()

    optimizer.zero_grad()
    last_train_step = total_train_step
    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr = lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss = step(input, target, optimizer_step=optimizer_step)

        it_time = time.time() - end

        if optimizer_step:
            if writer:
                writer.add_scalar("train/summary/scalar/learning_rate", lr, epoch)
                writer.add_scalar(
                    "train/summary/scalar/loss", to_python_float(loss), total_train_step
                )
                writer.add_scalar(
                    "perf/summary/scalar/compute_ips",
                    calc_ips(bs, it_time - data_time),
                    total_train_step,
                )
                writer.add_scalar(
                    "perf/summary/scalar/train_total_ips",
                    calc_ips(bs, it_time),
                    total_train_step,
                )
                mlflow.log_metric("train/learning_rate", step=epoch, value=lr)
                mlflow.log_metric(
                    "train/loss", step=total_train_step, value=to_python_float(loss)
                )
                mlflow.log_metric(
                    "perf/compute_ips",
                    step=total_train_step,
                    value=calc_ips(bs, it_time - data_time),
                )
                mlflow.log_metric(
                    "perf/train_total_ips",
                    step=total_train_step,
                    value=calc_ips(bs, it_time),
                )

            total_train_step += 1
        if logger is not None:
            logger.log_metric("train.loss", to_python_float(loss), bs)
            logger.log_metric("train.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("train.total_ips", calc_ips(bs, it_time))
            logger.log_metric("train.data_time", data_time)
            logger.log_metric("train.compute_time", it_time - data_time)

        end = time.time()

        if writer:
            writer.flush()

        if detector.is_preempted():
            print(
                datetime.utcnow(),
                "Exit training loop detecting is_preempted changed to True",
            )
            return last_train_step

    return total_train_step


def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(
    val_loader,
    model_and_loss,
    fp16,
    logger,
    epoch,
    detector,
    prof=-1,
    register_metrics=True,
):
    print(f"validating...")
    print(f"register_metrics {register_metrics}, logger {logger}.")
    if register_metrics and logger is not None:
        logger.register_metric(
            "val.top1",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.top5",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "val.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at100",
            log.LAT_100(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at99",
            log.LAT_99(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at95",
            log.LAT_95(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_val_step(model_and_loss)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    loss_sum = 0.0
    total_val_step = 0
    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)
        if logger is not None:
            logger.log_metric("val.top1", to_python_float(prec1), bs)
            logger.log_metric("val.top5", to_python_float(prec5), bs)
            logger.log_metric("val.loss", to_python_float(loss), bs)
            logger.log_metric("val.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("val.total_ips", calc_ips(bs, it_time))
            logger.log_metric("val.data_time", data_time)
            logger.log_metric("val.compute_latency", it_time - data_time)
            logger.log_metric("val.compute_latency_at95", it_time - data_time)
            logger.log_metric("val.compute_latency_at99", it_time - data_time)
            logger.log_metric("val.compute_latency_at100", it_time - data_time)

        loss_sum += to_python_float(loss)
        total_val_step += 1

        end = time.time()
        if detector.is_preempted():
            print(
                datetime.utcnow(),
                "Exit validation loop detecting is_preempted changed to True",
            )
            break

    return [top1, loss_sum / total_val_step]


# Train loop {{{
def calc_ips(batch_size, time):
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    tbs = world_size * batch_size
    return tbs / time


def train_loop(
    model_and_loss,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    epochs,
    fp16,
    logger,
    should_backup_checkpoint,
    save_checkpoint_epochs,
    use_amp=False,
    batch_size_multiplier=1,
    best_prec1=0,
    start_epoch=0,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    total_train_step=0,
):
    is_first_rank = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    if is_first_rank:
        ts = str(time.time())
        # logdir = os.path.expanduser('~/tensorboard/{}/logs/'.format(os.environ['DLTS_JOB_ID']) + ts)
        logdir = os.path.expanduser(
            "~/tensorboard/{}/logs/".format(os.environ["AZUREML_RUN_ID"]) + ts
        )
        print("tensorboard at ", logdir)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = None

    prec1 = -1
    detector = PreemptDetector()
    detector.run()

    epoch_iter = range(start_epoch, epochs)
    for epoch in epoch_iter:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        if writer:
            writer.add_scalar("train/summary/scalar/world_size", world_size, epoch)
            mlflow.log_metric("train/world_size", step=epoch, value=world_size)

        if logger is not None:
            logger.start_epoch()
        if not skip_training:
            total_train_step = train(
                train_loader,
                model_and_loss,
                optimizer,
                lr_scheduler,
                fp16,
                logger,
                epoch,
                detector,
                use_amp=use_amp,
                prof=prof,
                register_metrics=epoch == start_epoch,
                batch_size_multiplier=batch_size_multiplier,
                total_train_step=total_train_step,
                writer=writer,
            )

        if not skip_validation and not detector.is_preempted():
            top1, val_loss = validate(
                val_loader,
                model_and_loss,
                fp16,
                logger,
                epoch,
                detector,
                prof=prof,
                register_metrics=epoch == start_epoch,
            )
            if not detector.is_preempted():
                prec1, nimg = top1.get_val()
                if writer:
                    writer.add_scalar("val/summary/scalar/loss", val_loss, epoch)
                    writer.add_scalar("val/summary/scalar/prec1", prec1, epoch)
                    mlflow.log_metric("val/loss", step=epoch, value=val_loss)
                    mlflow.log_metric("val/prec1", step=epoch, value=prec1)

        if logger is not None:
            print(
                "Epoch ", epoch, " complete with is_preempted ", detector.is_preempted()
            )
            logger.end_epoch()

        save_ckpt = is_first_rank and (
            detector.is_preempted() or (epoch + 1) % save_checkpoint_epochs == 0
        )

        if detector.is_preempted() and start_epoch == epoch:
            print(
                "Skipping save checkpoint since no complete epoch finishes till now. ",
                start_epoch,
                "-->",
                epoch,
            )
            save_ckpt = False
        print(f"save ckpt {save_ckpt}, ckpt dir {checkpoint_dir}.")
        if save_ckpt:
            if not skip_validation and not detector.is_preempted():
                is_best = logger.metrics["val.top1"]["meter"].get_epoch() > best_prec1
                best_prec1 = max(
                    logger.metrics["val.top1"]["meter"].get_epoch(), best_prec1
                )
            else:
                is_best = False
                best_prec1 = 0

            ckpt_epoch_index = epoch + 1 if not detector.is_preempted() else epoch
            utils.save_checkpoint(
                {
                    "epoch": ckpt_epoch_index,
                    "arch": model_and_loss.arch,
                    "state_dict": model_and_loss.model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                    "total_train_step": total_train_step,
                },
                is_best,
                checkpoint_dir=checkpoint_dir,
                backup_filename=checkpoint_file_name,
            )

        if detector.is_preempted():
            print(
                datetime.utcnow(),
                "Exit epoch loop detecting is_preempted changed to True, save_ckpt:",
                save_ckpt,
            )
            break

    if writer:
        writer.close()
    detector.stop()
    print(
        datetime.utcnow(), "Training exits with is_preempted: ", detector.is_preempted()
    )


# }}}
