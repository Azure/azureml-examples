import argparse
import os
import shutil
import time

import torch
import torchvision
from azureml.core import Run
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm


LEARNING_RATE = 1e-2
MODEL_CHECKPOINT_PATH = "model_checkpoints/checkpoint.pt"


def get_device():
    if torch.cuda.is_available():
        return torch.device(0)
    return torch.device("cpu")


device = get_device()


def init_optimizer(model):
    return torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE)


# The following neural net, as well as some other components of this script, are from the
# PyTorch example tutorial for training on MNIST:
# https://github.com/pytorch/examples/blob/0352380e6c066ed212e570d6fe74e3674f496258/mnist/main.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def init_model():
    model = Net().to(device)
    return model


def train_epoch(model, train_data_loader, optimizer):
    for images, targets in tqdm(train_data_loader):

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = F.nll_loss(outputs, targets)

        loss.backward()
        optimizer.step()


def save_checkpoint(model, optimizer, epoch):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        MODEL_CHECKPOINT_PATH,
    )
    shutil.copy(MODEL_CHECKPOINT_PATH, "outputs/model.pt")


def score_validation_set(model, data_loader):
    print("\nEvaluating validation set accuracy...\n")

    with torch.no_grad():

        num_correct = 0
        num_total_images = 0

        for images, targets in tqdm(data_loader):

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            correct = torch.argmax(outputs, dim=1) == targets
            num_correct += torch.sum(correct).item()
            num_total_images += len(images)

        return num_correct, num_total_images


def load_checkpoint():
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
    model = init_model()
    optimizer = init_optimizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]


if __name__ == "__main__":

    torch.manual_seed(0)

    # Parse input command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs", type=int, help="total # of epochs to train the model"
    )
    args, _ = parser.parse_known_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    run = Run.get_context()

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(
        ".", train=True, download=True, transform=transform
    )
    valid_dataset = torchvision.datasets.MNIST(
        ".", train=False, download=True, transform=transform
    )

    checkpoint_file_exists = os.path.exists(MODEL_CHECKPOINT_PATH)
    print(f"Checkpoint file exists: {checkpoint_file_exists}")

    # Load state from saved checkpoints if saved checkpoints exist.
    # Otherwise, initialize a model from scratch.
    if checkpoint_file_exists:
        print("Loading saved checkpoints...")
        model, optimizer, starting_epoch = load_checkpoint()
        starting_epoch += 1
    else:
        model = init_model()
        optimizer = init_optimizer(model)
        starting_epoch = 0

    # Set number of workers to # of of CPUs
    num_workers = os.cpu_count()

    # Initialize data loaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=32, num_workers=num_workers, pin_memory=True
    )

    for epoch in range(starting_epoch, args.num_epochs):
        run.log("training_epoch", epoch)
        print(f"Starting epoch {epoch}")

        model.train()

        start = time.time()
        train_epoch(model, train_data_loader, optimizer)
        run.log("epoch_train_time", time.time() - start)

        run.flush()
        save_checkpoint(model, optimizer, epoch)

        model.eval()
        num_correct, num_total_images = score_validation_set(model, valid_data_loader)

        print(
            f"Scored validation set: {num_correct} correct, {num_total_images} total images"
        )
        validation_accuracy = num_correct / num_total_images * 100
        run.log("validation_accuracy", validation_accuracy)
        print(f"Accuracy: {validation_accuracy}%")

    print("Done")
