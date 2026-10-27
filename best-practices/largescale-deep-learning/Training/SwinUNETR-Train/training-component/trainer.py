# -------------------------------------------------------------------------
# Portions Copyright (c) Microsoft Corporation.  All rights reserved.
# --------------------------------------------------------------------------
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import mlflow
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, acc_func, post_label, post_pred, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.gpu), target.cuda(args.gpu)

        # Make sure the data is in the right shape
        if args.roi_x != data.shape[2] or args.roi_y != data.shape[3] or args.roi_z != data.shape[4]:
            data = torch.nn.functional.interpolate(data, size=(args.roi_x, args.roi_y, args.roi_z), mode='trilinear')
            target = torch.nn.functional.interpolate(target, size=(args.roi_x, args.roi_y, args.roi_z), mode='trilinear')

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if not logits.is_cuda:
            target = target.cpu()
        val_labels_list = decollate_batch(target)
        val_labels_convert = [
            post_label(val_label_tensor) for val_label_tensor in val_labels_list
        ]
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [
            post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]
        acc_func.reset()
        acc_func(y_pred=val_output_convert, y=val_labels_convert)

        acc, not_nans = acc_func.aggregate()
        acc = acc.cuda(args.gpu)

        if args.distributed:
            acc_list, not_nans_list = distributed_all_gather(
                [acc, not_nans],
                out_numpy=True,
                is_valid=idx < loader.sampler.valid_length,
            )
            acc_item = 0
            for al, nl in zip(acc_list, not_nans_list):
                run_acc.update(al, n=nl)
                acc_item += al.item()
            acc_item /= len(acc_list)
        else:
            acc_item = acc.item()
            run_acc.update(acc_item, n=not_nans.cpu().numpy())
        if args.rank == 0:
            mlflow.log_metric(
                "train_acc_running", acc_item, step=epoch * len(loader) + idx
            )

        if args.distributed:
            loss_list = distributed_all_gather(
                [loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length
            )
            loss_item = np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0)
            run_loss.update(
                loss_item,
                n=args.batch_size * args.world_size,
            )
        else:
            loss_item = loss.item()
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
            mlflow.log_metric(
                "train_loss_running", loss_item, step=epoch * len(loader) + idx
            )

        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg, run_acc.avg


def val_epoch(
    model,
    loader,
    epoch,
    loss_func,
    acc_func,
    post_label,
    post_pred,
    args,
    model_inferer=None,
):
    model.eval()
    run_loss = AverageMeter()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            loss = loss_func(logits, target)
            if args.distributed:
                loss_list = distributed_all_gather(
                    [loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                loss_item = np.mean(
                    np.mean(np.stack(loss_list, axis=0), axis=0), axis=0
                )
                run_loss.update(loss_item, n=args.batch_size * args.world_size)
            else:
                loss_item = loss.item()
                run_loss.update(loss_item, n=args.batch_size)
            if args.rank == 0:
                mlflow.log_metric(
                    "val_loss_running", loss_item, step=epoch * len(loader) + idx
                )

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans],
                    out_numpy=True,
                    is_valid=idx < loader.sampler.valid_length,
                )
                acc_item = 0
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
                    acc_item += al.item()
                acc_item /= len(acc_list)

            else:
                acc_item = acc.item()
                run_acc.update(acc_item, n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
                mlflow.log_metric(
                    "val_acc_running", acc_item, step=epoch * len(loader) + idx
                )
            start_time = time.time()
    return run_loss.avg, run_acc.avg


def save_checkpoint(
    model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None
):
    state_dict = (
        model.state_dict() if not args.distributed else model.module.state_dict()
    )
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    post_label,
    post_pred,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss, val_avg_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            loss_func=loss_func,
            acc_func=acc_func,
            args=args,
            post_label=post_label,
            post_pred=post_pred,
        )
        if args.rank == 0:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", val_avg_acc, step=epoch)
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_loss, val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                loss_func=loss_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)
            val_avg_loss = np.mean(val_avg_loss)

            if args.rank == 0:
                mlflow.log_metric("val_loss", val_avg_loss, step=epoch)
                mlflow.log_metric("val_acc", val_avg_acc, step=epoch)
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print(
                        "new best ({:.6f} --> {:.6f}). ".format(
                            val_acc_max, val_avg_acc
                        )
                    )
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if (
                        args.rank == 0
                        and args.logdir is not None
                        and args.save_checkpoint
                    ):
                        save_checkpoint(
                            model,
                            epoch,
                            args,
                            best_acc=val_acc_max,
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(
                    model, epoch, args, best_acc=val_acc_max, filename="model_final.pt"
                )
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(
                        os.path.join(args.logdir, "model_final.pt"),
                        os.path.join(args.logdir, "model.pt"),
                    )

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
