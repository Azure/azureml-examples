import argparse
import os

import mlflow
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


BATCH_SIZE = 256
NUM_WORKERS = int(os.cpu_count() / 2)
CHECKPOINT_FILE_NAME = "cifar10-checkpoint"


class LitResnet(LightningModule):
    def __init__(self, data_dir, batch_size=256, learning_rate=1e-3, num_workers=4):
        super().__init__()

        self.save_hyperparameters()
        self.model = self.create_model()

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 10)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self._data_dir = data_dir
        self._num_workers = num_workers
        self._train_dataloader = None
        self._val_dataloader = None

    def create_model(self):
        model = models.resnet18(pretrained=False, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        return model

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def accuracy(self, preds, y):
        correct = 0
        total = 0
        for pred, label in zip(preds, y):
            if pred == label:
                correct += 1
            total += 1
        acc = correct / total
        return acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def train_dataloader(self):
        if self._train_dataloader is None:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                    ),
                ]
            )

            trainset = datasets.CIFAR10(root=self._data_dir, train=True, download=False, transform=transform)
            self._train_dataloader = DataLoader(
                trainset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self._num_workers,
            )
        return self._train_dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                    ),
                ]
            )

            valset = datasets.CIFAR10(root=self._data_dir, train=False, download=False, transform=transform)
            self._val_dataloader = DataLoader(
                valset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self._num_workers,
            )
        return self._val_dataloader

    def on_train_epoch_end(self):
        self.log("world_size", int(os.environ.get("GROUP_WORLD_SIZE", 1)))

    def on_train_end(self):
        if int(os.environ["RANK"]) == 0:
            print(f"Saving model to {args.output_dir}/model.pt")
            mlflow.pytorch.log_model(self.model, f"{args.output_dir}/model")
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, "model.pt")
            torch.save(self.model, model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory_name", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--experiment_name", required=False, default=None, type=str)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num_nodes", required=False, default=1, type=int)
    parser.add_argument("--num_devices", required=False, default=1, type=int)
    parser.add_argument("--strategy", required=False, default="ddp")
    parser.add_argument("--dist_backend", required=False, default="nccl")
    args = parser.parse_args()

    return args


def main(args):
    print(f"is_elastic_launched: {torch.distributed.is_torchelastic_launched()}")
    num_nodes = args.num_nodes
    num_devices = args.num_devices
    if torch.distributed.is_torchelastic_launched():
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", 1))
        num_devices = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        print(
            f"TorchElastic: world_size: {world_size}, rank: {rank}, local_rank: {local_rank}, num_nodes: {num_nodes}"
        )

        print(
            f"Initalizing process group with {args.dist_backend} backend, \
                rank {rank}, world_size {world_size}"
        )
        torch.distributed.init_process_group(backend=args.dist_backend)

    model = LitResnet(
        data_dir=args.dataset_directory_name,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=CHECKPOINT_FILE_NAME,
        every_n_epochs=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )
    mlflow_logger = MLFlowLogger(
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", args.experiment_name),
        log_model=False,
        run_id=os.getenv("MLFLOW_RUN_ID"),
        artifact_location=args.output_dir,
    )

    trainer = Trainer(
        default_root_dir="outputs",
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=num_devices,
        num_nodes=num_nodes,
        strategy=args.strategy,
        logger=mlflow_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
        ],
    )

    ckpt_path = None
    if args.checkpoint_dir:
        print(f"Loading checkpoint from {args.checkpoint_dir}/{CHECKPOINT_FILE_NAME}.ckpt")
        if os.path.isfile(f"{args.checkpoint_dir}/{CHECKPOINT_FILE_NAME}.ckpt"):
            ckpt_path = f"{args.checkpoint_dir}/{CHECKPOINT_FILE_NAME}.ckpt"
            print(f"Loading checkpoint from {ckpt_path}")

    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
