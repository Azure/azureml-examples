# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse

import lightning.pytorch as pl

from timm import create_model
from torch import optim, nn, utils
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def make_dataset(root_folder, subset):
    if subset == "train":
        train_dataset = ImageFolder(
            root_folder + "/train",
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        return train_dataset

    if subset == "val":
        val_dataset = ImageFolder(
            root_folder + "/val",
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        return val_dataset

    return None


class ViTLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=1000, img_size=224
        )

        self.accuracy = Accuracy(task="multiclass", num_classes=1000)

    def training_step(self, batch, _):
        # Get the images and labels.
        x, y = batch

        # Compute the training loss.
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        # Log the training loss.
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        # Get the images and labels.
        x, y = batch

        # Compute the validation loss.
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        # Log the validation loss.
        self.log("val_loss", loss, sync_dist=True)

        # Log and print validation accuracy.
        self.accuracy(y_hat, y)
        self.log(
            "val_acc",
            self.accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        # Make the optimizer and learning rate scheduler.
        optimizer = optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=False
        )
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory_name", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", required=False, default=1, type=int)
    parser.add_argument("--num_nodes", required=False, default=1, type=int)
    parser.add_argument("--num_devices", required=False, default=1, type=int)
    parser.add_argument("--strategy", required=False, default="ddp")
    args = parser.parse_args()

    return args


def launch_training(args):
    print("creating training set...")
    train_dataset = make_dataset(args.dataset_directory_name, "train")
    train_loader = utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("done")

    print("creating validation set...")
    val_dataset = make_dataset(args.dataset_directory_name, "val")
    val_loader = utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("done")

    print("creating model...")
    model = ViTLightning()
    print("done")

    print("launching training")
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        devices=args.num_devices,
        strategy=args.strategy,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    args = parse_args()

    print("running image classification...")
    launch_training(args)
    print("done")
