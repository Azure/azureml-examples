# Copyright (c) 2017 Facebook, Inc. All rights reserved.
# BSD 3-Clause License
#
# Script adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# ==============================================================================

# imports
import os
import mlflow
import argparse
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from azure.monitor.opentelemetry import configure_azure_monitor
configure_azure_monitor(
    connection_string="InstrumentationKey=f5f6034c-07db-4eb2-8807-c309cfc5cbe1;IngestionEndpoint=https://eastus-3.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/;ApplicationId=db740992-fe57-44cb-a7b4-e14ec334181e",
)
from opentelemetry import metrics
import os
from azureml.core import Run
meter = metrics.get_meter_provider().get_meter("custom_training_metrics")
histogram = meter.create_histogram("loss")
run = Run.get_context()
workspace = run.experiment.workspace

# TODO - add mlflow logging

# define network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 6 * 6, 120)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def top1_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch."""
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct / targets.size(0)

def get_lr(optimizer) -> float:
    """Get current LR from the first param group."""
    return optimizer.param_groups[0].get("lr", None)

# define functions
def train(train_loader, model, criterion, optimizer, epoch, device, print_freq, rank, distributed):
    running_loss = 0.0
    epoch_start = time.time()
    epoch_loss_sum = 0.0
    epoch_examples = 0
    epoch_acc_sum = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Accumulate epoch stats
        batch_size = inputs.size(0)
        epoch_examples += batch_size
        epoch_loss_sum += loss.item() * batch_size
        epoch_acc_sum += top1_accuracy(outputs, labels) * batch_size

        # --- Per-iteration MLflow metric (rank 0 only) ---
        if (not distributed) or (rank == 0):
            # step = global iteration index; you can also use i or (epoch, i)
            global_step = epoch * len(train_loader) + i
            mlflow.log_metric("loss_iter", loss.item(), step=global_step)

        # --- Per-epoch metrics ---
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss_sum / max(epoch_examples, 1)
        avg_acc = epoch_acc_sum / max(epoch_examples, 1)
        lr_now = get_lr(optimizer)
        throughput = epoch_examples / max(epoch_time, 1e-6)

        # Rank 0 logs
        if (not distributed) or (rank == 0):
            mlflow.log_metrics({
                "loss_epoch": avg_loss,
                "acc_epoch_top1": avg_acc,
                "throughput_img_per_sec": throughput,
                "epoch_time_sec": epoch_time,
            }, step=epoch)

        print(f"Rank {rank}: Finished epoch {epoch} | "
              f"loss={avg_loss:.4f} acc@1={avg_acc:.4f} lr={lr_now} "
              f"thr={throughput:.1f} img/s")

        # print statistics
        running_loss += loss.item()
        if i % print_freq == 0:  # print every print_freq mini-batches
            histogram.record(running_loss / print_freq, {
                "sub_id": workspace.subscription_id, "run_id": run.id,
                "workspace": workspace.name, "resource_group": workspace.resource_group})
            #print(
            #    "Rank %d: [%d, %5d] loss: %.3f"
            #    % (rank, epoch + 1, i + 1, running_loss / print_freq)
            #)
            running_loss = 0.0


def main(args):
    # get PyTorch environment variables
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    distributed = world_size > 1

    # set device
    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize distributed process group using default env:// method
    if distributed:
        torch.distributed.init_process_group(backend="nccl")

    # define train and dataset DataLoaders
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
    )

    model = Net().to(device)

    # wrap model with DDP
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

    
    if (not distributed) or (rank == 0):
        mlflow.start_run()
        # Log static params once
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "world_size": world_size,
            "distributed": distributed,
            "model": "Net(CIFAR10)",
        })


    # train the model
    for epoch in range(args.epochs):
        print("Rank %d: Starting epoch %d" % (rank, epoch))
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            args.print_freq,
            rank,
            distributed,
        )

    print("Rank %d: Finished Training" % (rank))

    if not distributed or rank == 0:
        # log model
        mlflow.pytorch.log_model(model, "model")
        os.makedirs(args.model_dir, exist_ok=True)
        torch.save(model, os.path.join(args.model_dir, "model.pt"))
        # mlflow.pytorch.save_model(model, f"{args.model_dir}/model")


def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--data-dir", type=str, help="directory containing CIFAR-10 dataset"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./", help="output directory for model"
    )
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="mini batch size for each gpu/process",
    )
    parser.add_argument(
        "--workers",
        default=2,
        type=int,
        help="number of data loading workers for each gpu/process",
    )
    parser.add_argument(
        "--learning-rate", default=0.001, type=float, help="learning rate"
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument(
        "--print-freq",
        default=200,
        type=int,
        help="frequency of printing training statistics",
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # call main function
    main(args)