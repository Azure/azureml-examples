# Copyright (c) 2017 Facebook, Inc. All rights reserved.
# BSD 3-Clause License
#
# Script adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# ==============================================================================

# imports
import argparse
import copy
import os

import mlflow
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#- add mlflow logging
mlflow.set_experiment("pipeline_samples") # enter the corresponding experiment name

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
def train(train_loader, valid_loader,  model, criterion, optimizer, epoch, device, print_freq, rank):
    running_loss = 0.0
    epoch_loss=0.0
    model.train()
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

        # print statistics
        running_loss += loss.item()
        epoch_loss+= loss.item()
        if i % print_freq == 0:  # print every print_freq mini-batches
            print(
                "Rank %d: [%d, %5d] loss: %.3f"
                % (rank, epoch + 1, i + 1, running_loss / print_freq)
            )

            running_loss = 0.0
    model.eval()
    valid_loss=0.0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # print statistics
            valid_loss += loss.item()
        print("Rank %d: [%d] valid loss: %.3f" % (rank, epoch + 1, valid_loss /  len(valid_loader) ))
        
    return epoch_loss/ len(train_loader), valid_loss /  len(valid_loader)
            


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

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform
    )
    n_train= len(trainset)
    # split train set into trainset and validset
    # select 20% as valid and the rest as training
    trains = list(np.random.choice(list(range(0, n_train)), size= int(0.8*n_train), replace=False))
    valids = [ idd for idd in list(range(0, n_train)) if idd not in trains]
    train_set = torch.utils.data.Subset(trainset, trains)
    valid_set = torch.utils.data.Subset(trainset, valids)


    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set)
    else:
        train_sampler = None
        valid_sampler = None
    

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
    )
    
    
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
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

    # train the model
    valid_min_loss= float('inf')
    best_model= copy.deepcopy(model)
    with mlflow.start_run() as run:
        for epoch in range(args.epochs):
            print("Rank %d: Starting epoch %d" % (rank, epoch))
            if distributed:
                train_sampler.set_epoch(epoch)
            model.train()
            # Start the run, log metrics, end the run

            train_loss, valid_loss=train(
                train_loader,
                valid_loader,
                model,
                criterion,
                optimizer,
                epoch,
                device,
                args.print_freq,
                rank,
            )
            mlflow.log_metric('train loss',train_loss, step=epoch )
            mlflow.log_metric('validation loss',valid_loss, step=epoch )
            
            if valid_min_loss>=valid_loss:
                valid_min_loss=valid_loss
                print("Rank %d: Found min valid loss" % (rank))
                best_model=copy.deepcopy(model)
                
    print("Rank %d: Finished Training" % (rank))
    if not distributed or rank == 0:
        # log model
        print("Rank %d: Found min valid loss, saving the model..." % (rank))
        mlflow.pytorch.save_model(best_model, f"{args.model_dir}/model")
        print("done.")



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
