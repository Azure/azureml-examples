import argparse
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# This defines the layers of the Neural Net?
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Added the view for reshaping score requests
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            # Use MLflow logging
            mlflow.log_metric("epoch_loss", loss.item())


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\n")
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(
                test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    # Use MLflow logging
    mlflow.log_metric("average_loss", test_loss)
    mlflow.log_metric("accuracy", 100.0 * correct / len(test_loader.dataset))


def get_data_loaders(trainds, testds, use_cuda=False, batch_size=64, test_batch_size=1000):
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        trainds,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        testds,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs
    )
    return train_loader, test_loader


def get_mnist_default_datasets():
    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=data_transforms)
    test = datasets.MNIST(
        "../data",
        train=False,
        transform=data_transforms)

    return train, test


def get_mnist_labeled_datasets(labeled_dataset):
    from azureml.core import Run, Dataset
    from azureml.dataprep.rslex import BufferingOptions, Downloader, CachingOptions
    from azureml_dataset import AzureMLDataset

    run = Run.get_context()
    ws = run.experiment.workspace
    labeled_dset = Dataset._load(labeled_dataset, ws)

    train, test = labeled_dset.random_split(0.9)

    caching_options = CachingOptions(512 * 1024 * 1024, None)
    downloader = Downloader(1024, 8, caching_options)
    buffering_options = BufferingOptions(1, downloader)
    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train = AzureMLDataset(train, buffering_options=buffering_options,
                           data_transforms=data_transforms, label_transforms=int)
    test = AzureMLDataset(test, buffering_options=buffering_options,
                          data_transforms=data_transforms, label_transforms=int)

    return train, test


def train_model(device, use_cuda, train_dataset, test_dataset, batch_size=64, test_batch_size=1000, epochs=3, lr=0.01, momentum=0.5, log_interval=10):
    train_loader, test_loader = get_data_loaders(
        train_dataset,
        test_dataset,
        use_cuda,
        batch_size,
        test_batch_size)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)

    return model


def main():
    # warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--labeled-dataset', type=str, default=None)
    parser.add_argument('--model-dir', type=str, default=None)
    args = parser.parse_args()

    mlflow.autolog()

    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.labeled_dataset:
        train_dataset, test_dataset = get_mnist_labeled_datasets(
            args.labeled_dataset)
    else:
        train_dataset, test_dataset = get_mnist_default_datasets()

    model = train_model(device, use_cuda, train_dataset, test_dataset, args.batch_size,
                        args.test_batch_size, args.epochs, args.lr, args.momentum, args.log_interval)

    # Log model to run history using MLflow
    mlflow.pytorch.log_model(
        model,
        "model",
        # registered_model_name="mnist-model"
    )


if __name__ == "__main__":
    main()
