# Copyright (c) 2017 Facebook, Inc. All rights reserved.
# BSD 3-Clause License
#
# Script adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# ==============================================================================
# imports
import datetime
import logging
import time

import mlflow
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import os
import argparse
import shutil
import socket


MODEL_CHECKPOINT_DIR = "outputs/model_checkpoints"
MODEL_CHECKPOINT_PATH = f"{MODEL_CHECKPOINT_DIR}/checkpoint.pt"


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


# define functions
def train(train_loader, model, criterion, optimizer, epoch, device, print_freq, rank):
    running_loss = 0.0
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

        if rank == 0:
            # compute gradient norm
            grad_norm = 0.0
            _model = model.module if hasattr(model, "module") else model
            for param in _model.parameters():
                grad_norm += torch.norm(param.grad).item()

            mlflow.log_metric("training_loss", loss.item(), step=i)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=i)
            mlflow.log_metric("gradient_norm", grad_norm, step=i)

            # print statistics
            running_loss += loss.item()
            if i % print_freq == 0:  # print every print_freq mini-batches
                print("Rank %d: [%d, %5d] loss: %.3f" % (rank, epoch + 1, i + 1, running_loss / print_freq))
                running_loss = 0.0


def evaluate(test_loader, model, device):
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    model.eval()

    correct = 0
    total = 0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # print total test set accuracy
    print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

    # print test accuracy for each of the classes
    for i in range(10):
        print("Accuracy of %5s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))

    mlflow.log_metric("validation_accuracy", 100 * correct / total)


def save_checkpoint(model, optimizer, epoch):
    # create checkpoint directory if it doesn't exist
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        MODEL_CHECKPOINT_PATH,
    )
    shutil.copy(MODEL_CHECKPOINT_PATH, "outputs/model.pt")


def load_checkpoint(args, device):
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
    model = init_model(device)
    optimizer = init_optimizer(model, args)
    state_dict = checkpoint["model_state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]


def init_model(device):
    model = Net().to(device)
    return model


def init_optimizer(model, args):
    return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)


def main(args):
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print(f"IP: {ip}")

    # get PyTorch environment variables
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"world_size: {world_size} rank: {rank} local_rank: {local_rank}")

    distributed = world_size > 1

    # set device
    if distributed:
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize distributed process group using default env:// method
    if distributed:
        torch.distributed.init_process_group(
            backend=args.dist_backend, world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=60)
        )

    # define train and test dataset DataLoaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    checkpoint_file_exists = os.path.exists(MODEL_CHECKPOINT_PATH)
    print(f"Checkpoint file exists: {checkpoint_file_exists}")

    # Load state from saved checkpoints if saved checkpoints exist.
    # Otherwise, initialize a model from scratch.
    if checkpoint_file_exists:
        print("Loading saved checkpoints...")
        model, optimizer, starting_epoch = load_checkpoint(args, device)
        starting_epoch += 1
    else:
        model = init_model(device)

        # wrap model with DDP
        if distributed:
            if not torch.cuda.is_available():
                model = nn.parallel.DistributedDataParallel(model)
            else:
                model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        # define optimizer
        optimizer = init_optimizer(model, args)
        starting_epoch = 0

    print(f"Creating dataloaders using data at {args.data_dir}...")
    train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform)

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

    test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # log model and optimizer parameters to mlflow
    if rank == 0:
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("momentum", args.momentum)

    # train the model
    for epoch in range(starting_epoch, args.epochs):
        print("Rank %d: Starting epoch %d" % (rank, epoch))
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        start = time.time()
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            args.print_freq,
            rank,
        )
        print(f"Epoch train time: {time.time() - start}")
        save_checkpoint(model, optimizer, epoch)

    print("Rank %d: Finished Training" % (rank))

    if not distributed or rank == 0:
        mlflow.pytorch.log_model(model, "model")
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "model.pt")
        torch.save(model, model_path)

        # evaluate on full test dataset
        evaluate(test_loader, model, device)


# run script
if __name__ == "__main__":
    # setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", type=str, help="directory containing CIFAR-10 dataset")
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="mini batch size for each gpu/process",
    )
    parser.add_argument(
        "--workers",
        default=2,
        type=int,
        help="number of data loading workers for each gpu/process",
    )
    parser.add_argument("--learning-rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--output-dir", default="outputs", type=str, help="directory to save model to")
    parser.add_argument(
        "--print-freq",
        default=200,
        type=int,
        help="frequency of printing training statistics",
    )
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        choices=["nccl", "gloo"],
        type=str,
        help="distributed backend",
    )
    args = parser.parse_args()

    # call main function
    main(args)
