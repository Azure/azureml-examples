import os
import numpy as np
from torch.utils import data
from torch import nn
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time


class feature_loader(data.Dataset):
    def __init__(self, data_dict, csv, mode="train"):
        self.data_dict = data_dict
        self.csv = csv
        self.mode = mode
        self.img_name = data_dict["img_name"]
        self.features = data_dict["features"]

    def __getitem__(self, item):
        img_name = self.img_name[item]
        features = self.features[item]
        features = features.astype("float32")

        row = self.csv[self.csv["Name"] == img_name]
        if self.mode == "train" or self.mode == "val":
            label = row["Label"].values

            label = np.array(label)
            label = np.reshape(label, (1,))
            label = label.squeeze()

            return features, label, img_name

        elif self.mode == "test":
            return features, img_name

    def __len__(self):
        return len(self.img_name)


## MLP Adaptors
## Input: 1-Dimensional Embeddings
## in_channels: Number of channels for input embeddings, num_class: Number of classes, finetune_mode: image (image-only)
## Output: Class-wise Prediction
class MLP_model(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_class):
        super().__init__()

        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_class = num_class

        ## Adaptor Module
        self.vision_embd = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

        self.retrieval_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=512,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
        )

        ## Prediction Head
        self.prediction_head = nn.Sequential(nn.Linear(self.hidden_dim, self.num_class))

    def forward(self, vision_feat):
        feat = self.vision_embd(vision_feat.squeeze(1))
        feat = self.retrieval_conv(torch.unsqueeze(feat, 2))
        class_output = self.prediction_head(feat.squeeze(2))

        return feat, class_output


def load_label_csv(label_csv_file):
    """
    Loads the label CSV file.

    Args:
    - label_csv_file (str): Path to the label CSV file.

    Returns:
    - df_label (pandas.DataFrame): Loaded label CSV as a DataFrame.
    """
    df_label = pd.read_csv(label_csv_file)
    return df_label


def create_data_loader(samples, csv, mode, batch_size, num_workers=2, pin_memory=True):
    """
    Creates a data loader for the generated embeddings.

    Args:
    - samples (dict): Dictionary containing the features and image names.
    - csv (pandas.DataFrame): DataFrame containing the labels.
    - mode (str): Mode of the data loader (train or test).
    - batch_size (int): Batch size for the data loader.
    - num_workers (int): Number of workers for the data loader (default: 2).
    - pin_memory (bool): Whether to pin the memory for the data loader (default: True).

    Returns:
    - data_loader (torch.utils.data.DataLoader): Data loader for the generated embeddings.
    """
    ds = feature_loader(samples, csv=csv, mode=mode)
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader


def create_output_directory(output_dir):
    """
    Create the output directory if it does not exist.

    Args:
    - output_dir (str): Path to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def create_model(in_channels, hidden_dim, num_class):
    """
    Create a model for the adaptor model (Default: MLP).

    Args:
        in_channels (int): Number of input channels.
        hidden_dim (int): Dimension of the hidden layer.
        num_class (int): Number of output classes.

    Returns:
        torch.nn.Module: The created MLP model.
    """
    model = MLP_model(
        in_channels=in_channels, hidden_dim=hidden_dim, num_class=num_class
    )
    return model


def trainer(train_ds, test_ds, model, loss_function_ts, optimizer, epochs, root_dir):
    """
    Trains a classification model and evaluates it on a validation set.
    Saves the model with the best validation ROC AUC score.
    """

    start_time = time.time()

    max_epoch = epochs
    best_metric = -1
    best_acc = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(max_epoch):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{max_epoch}")
        model.train()
        epoch_loss = 0
        step = 0

        # Training loop
        for batch_idx, (features, pathology_label, img_name) in tqdm(
            enumerate(train_ds),
            total=len(train_ds),
            desc=f"Train Epoch={epoch}",
            ncols=80,
            leave=False,
        ):

            step += 1
            features = features.to(device)
            pathology_label = pathology_label.to(device)

            optimizer.zero_grad()
            _, pred_pathology = model(features)

            loss = loss_function_ts(pred_pathology, pathology_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print(f"{step}/{len(train_ds)}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            y_pred_list = []
            y_true_list = []

            for batch_idx, (features, pathology_label, img_name) in tqdm(
                enumerate(test_ds),
                total=len(test_ds),
                desc=f"Test Epoch={epoch}",
                ncols=80,
                leave=False,
            ):

                features = features.to(device)
                pathology_label = pathology_label.to(device)

                _, pred_pathology = model(features)

                y_pred_list.append(pred_pathology)
                y_true_list.append(pathology_label)

            # Concatenate predictions and true labels
            y_pred = torch.cat(y_pred_list, dim=0)
            y_true = torch.cat(y_true_list, dim=0)

            # Compute probabilities for the positive class
            y_scores = torch.softmax(y_pred, dim=1).cpu().numpy()
            y_true_np = y_true.cpu().numpy()

            # Compute ROC AUC
            auc = roc_auc_score(y_true_np, y_scores, multi_class="ovr")

            # Compute accuracy
            acc_metric = (y_pred.argmax(dim=1) == y_true).sum().item() / len(y_true)

            metric_values.append(auc)

            # Save the best model
            if auc > best_metric:
                best_metric = auc
                best_acc = acc_metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print("Saved new best metric model")

            print(
                f"Current epoch: {epoch + 1} Current AUC: {auc:.4f}"
                f" Current accuracy: {acc_metric:.4f}"
                f" Best AUC: {best_metric:.4f}"
                f" Best accuracy: {best_acc:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

    end_time = time.time()
    training_time = end_time - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Training Time: {int(hours):02}:{int(minutes):02}:{seconds:.2f}")
    print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    return best_acc, best_metric


def perform_inference(model, test_loader):
    predictions = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for features, img_names in tqdm(test_loader, desc="Inference", ncols=80):
            features = features.to(device)
            _, output = model(features)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1)
            predicted_classes = probabilities.argmax(dim=1).cpu().numpy()
            # Collect predictions
            for img_name, predicted_class, prob in zip(
                img_names, predicted_classes, probabilities.cpu().numpy()
            ):
                predictions.append(
                    {
                        "Name": img_name,
                        "PredictedClass": predicted_class,
                        "Probability": prob[predicted_class],
                    }
                )
    return predictions


def load_trained_model(model, model_path):
    # Load Model State
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model
