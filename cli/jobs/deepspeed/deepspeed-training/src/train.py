# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Train a model using deepspeed."""
import argparse
import os
import deepspeed
import torch
import torchvision
import torchvision.transforms as transforms
import time
from model import Net, nn

# import MLflow if available. Continue with a warning if not installed on the system.
try:
    import mlflow
except ImportError:
    print("MLFlow logging failed. Continuing without MLflow.")
    pass


def add_argument():
    """Add arguements for deepspeed."""
    parser = argparse.ArgumentParser(description="CIFAR")

    parser.add_argument(
        "--data-dir", type=str, help="directory containing CIFAR-10 dataset"
    )

    # train
    parser.add_argument(
        "-b", "--batch_size", default=32, type=int, help="mini-batch size (default: 32)"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=3,
        type=int,
        help="number of total epochs (default: 3)",
    )

    # distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("--global_rank", default=-1, type=int, help="global rank")
    parser.add_argument(
        "--with_aml_log", default=True, help="Use Azure ML metric logging"
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


# Need args here to set ranks for multi-node training with download=True
args = add_argument()
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

data_files = os.listdir(args.data_dir)
expected_file = "cifar-10-batches-py"
if expected_file not in data_files:
    print("Folder {} expected in args.data_dir".format(expected_file))
    print("Found:")
    print(data_files)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root=args.data_dir, train=True, download=False, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root=args.data_dir, train=False, download=False, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

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

########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

net = Net()
parameters = filter(lambda p: p.requires_grad, net.parameters())


# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset
)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

criterion = nn.CrossEntropyLoss()

# DeepSpeed optimizer is already set so no need to use the following
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# Showcasing logging metrics to automl.
if args.with_aml_log:
    this_run = mlflow.active_run()
    print("Active run_id: {}".format(this_run.info.run_id))
    mlflow.log_metrics({"hello": 12345})
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.
for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        pre = time.time()
        inputs, labels = (
            data[0].to(model_engine.local_rank),
            data[1].to(model_engine.local_rank),
        )

        outputs = model_engine(inputs.half())
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()
        post = time.time()
        Time_per_step = post - pre

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:  # print every 2000 mini-batches
        loss = running_loss / 2000
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, loss))
        if args.with_aml_log:
            try:
                mlflow.log_metrics({"loss": loss})
            except NameError:
                print("MLFlow logging failed. Continuing without MLflow.")
                pass
        running_loss = 0.0

print("Finished Training")

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images.to(model_engine.local_rank).half())

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(model_engine.local_rank).half())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(model_engine.local_rank)).sum().item()

print(
    "Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total)
)

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(model_engine.local_rank).half())
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(model_engine.local_rank)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(
        "Accuracy of %5s : %2d %%"
        % (classes[i], 100 * class_correct[i] / class_total[i])
    )
