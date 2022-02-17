"""
TODO: checkpoint path as input
"""
from __future__ import print_function, division
import argparse
import time
import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from distutils.util import strtobool
import csv
import logging
import warnings
from typing import Any, Callable, List, Optional, Tuple
#warnings.filterwarnings("ignore")
import glob



def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    dataloaders,
    dataset_sizes,
    class_names,
    device,
):
    """
    Train the model
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "valid"]:
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                corrects = torch.sum(preds == labels.data).float()
                running_corrects += corrects

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class ImageDatasetWithLabelInMap(torchvision.datasets.VisionDataset):
    """PyTorch dataset for images in a folder, with label provided as a dict."""
    def __init__(self, root: str, image_labels: dict, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        # calling VisionDataset.__init__() first
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)

        # now the specific initialization
        self.samples = [] # list of tuples (path,target)
        
        # search for all images
        images_in_root = glob.glob(root+"/*")

        # find their target
        for entry in images_in_root:
            entry_basename = os.path.basename(entry)
            if entry_basename not in image_labels:
                logging.warning(f"Image in root dir {entry} is not in provided image_labels")
            else:
                self.samples.append(
                    (entry, 1 if image_labels[entry_basename]=="contains_person" else 0)
                )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class PyTorchImageModelTraining:
    def __init__(self):
        self.model = None
        self.output_classes = []
        self.training_params = {}
        self.training_labels = {}
        self.validation_labels = {}

        self.training_data_loader = None
        self.validation_data_loader = None

        self.logger = logging.getLogger(__name__)

    def load_image_labels(
        self, training_image_labels_path: str, validation_image_labels_path: str
    ):
        with open(training_image_labels_path, newline='') as csv_file:
            training_csv_reader = csv.reader(csv_file, delimiter=",")

            self.training_labels = []
            training_classes = set()
            for row in training_csv_reader:
                self.training_labels.append(
                    (os.path.basename(row[0]), row[1])
                )
                training_classes.add(row[1])
        self.training_labels = dict(self.training_labels)

        self.logger.info(f"Loaded training annotations, has samples={len(self.training_labels)} and classes={training_classes}")

        with open(validation_image_labels_path, newline='') as csv_file:
            validation_csv_reader = csv.reader(csv_file, delimiter=",")

            self.validation_labels = []
            validation_classes = set()
            for row in validation_csv_reader:
                if row[1] not in training_classes:
                    self.logger.warning(f"Validation image {row[0]} has class {row[1]} that is not in the training set classes {training_classes}, this image will be discarded.")
                else:
                    self.validation_labels.append(
                        (os.path.basename(row[0]), row[1])
                    )
                    validation_classes.add(row[1])
        self.validation_labels = dict(self.validation_labels)
        
        self.logger.info(f"Loaded validation annotations, has samples={len(self.validation_labels)} and classes={training_classes}")

        if validation_classes != training_classes:
            raise Exception(f"Validation classes {validation_classes} != training classes {training_classes}, we can't proceed with training.")

        self.output_classes = sorted(list(training_classes))


    def load_model(
        self,
        model_arch: str,
        checkpoint_path: str = None,
        output_classes: int = None,
        pretrained: bool = True,
    ):
        """Loads a model from a given arch and sets it up for training"""
        # Load pretrained resnet model
        if output_classes is None:
            output_classes = len(self.output_classes)

        self.logger.info(f"Loading model from arch={model_arch} pretrained={pretrained} output_classes={output_classes}")
        if model_arch == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, output_classes)
        else:
            raise NotImplementedError(
                f"model_arch={model_arch} is not implemented yet."
            )

        # Use CUDA if it is available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        self.model = self.model.to(device)

        return self.model


    def load_images(self, train_images_dir, valid_images_dir, batch_size):
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(200),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_dataset = ImageDatasetWithLabelInMap(
            root=train_images_dir,
            image_labels=self.training_labels,
            transform = train_transform
        )
        self.logger.info(f"ImageDatasetWithLabelInMap loaded training image list samples={len(train_dataset)}")

        self.training_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        valid_transform = transforms.Compose(
            [
                transforms.Resize(200),
                transforms.CenterCrop(200),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]),
            ]
        )
        valid_dataset = ImageDatasetWithLabelInMap(
            root=valid_images_dir,
            image_labels=self.validation_labels,
            transform = valid_transform
        )
        self.logger.info(f"ImageDatasetWithLabelInMap loaded validation image list samples={len(valid_dataset)}")
        self.validation_data_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )


    def train(self, learning_rate=5e-5, momentum=0.9):
        # Specify criterion
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=1e-4)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Train model
        model = train_model(
            resnet,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            dataloaders,
            dataset_sizes,
            class_names,
            device,
        )

        # Save model
        print("Saving model")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model, os.path.join(output_dir, "model.pt"))
        classes_file = open(os.path.join(output_dir, "class_names.pkl"), "wb")
        pickle.dump(class_names, classes_file)
        classes_file.close()


def main():
    """Main function of the script."""
    # initialize root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_images",
        type=str,
        required=True,
        help="path to folder containing training images",
    )
    parser.add_argument(
        "--valid_images",
        type=str,
        required=True,
        help="path to folder containing validation images",
    )
    parser.add_argument(
        "--train_annotations",
        type=str,
        required=True,
        help="readable name of the category",
    )
    parser.add_argument(
        "--valid_annotations",
        type=str,
        required=True,
        help="path to output train annotations",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="path to read and write checkpoints",
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        required=False,
        choices=["resnet18"],
        default="resnet18",
        help="which model architecture to use (default: resnet18)",
    )
    parser.add_argument(
        "--model_arch_pretrained",
        type=strtobool,
        required=False,
        default=True,
        help="use pretrained model (default: true)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=1,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="Train batch size (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=0.01,
        help="Learning rate of optimizer",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        required=False,
        default=0.01,
        help="Momentum of optimizer",
    )

    args = parser.parse_args()

    logger.info(f"Running with arguments: {args}")

    training_handler = PyTorchImageModelTraining()

    training_handler.load_image_labels(
        args.train_annotations,
        args.valid_annotations
    )

    training_handler.load_model(
        args.model_arch,
        checkpoint_path=args.checkpoint_path,
        pretrained=args.model_arch_pretrained,
    )

    training_handler.load_images(
        train_images_dir=args.train_images,
        valid_images_dir=args.valid_images,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
