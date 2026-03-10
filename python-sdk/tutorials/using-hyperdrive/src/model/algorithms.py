"""
Script that runs a calculation-based model with four outputs that 
mimic performance metrics for a hypothetical binary classifier.
Find more details about this model at the notebook named
Fictitious_Model_for_Hyperparameter_Tuning.ipynb in the parent directory. 
"""
import numpy as np


def train_fictitious_model(thr: float, epoch: int, lr: float) -> dict:
    """
    Parameters:
      thr: float: input that mimics threshold in a binary classification problem
      epoch: int: input that mimics number of epochs in a deep learning algorithm
      lr: float: input that mimics learning rate in a model learning algorithm
    Returns:
      result: dict: list of metrics calculated using input parameters
    """

    # Constant scalar factors
    k_a = 1
    k_p = 1
    k_r = 1
    k_l = 1

    # Outputs that mimic precision and recall in a binary classification problem as
    # functions of an input that mimics threshold
    precision_thr = 0.1 + 0.9 / (1 + 5000 * np.exp(-20 * thr))
    recall_thr = 1 / (1 + 1e-5 * np.exp(20 * thr))

    # Outputs that mimic training loss and model accuracy as functions of an input
    # that mimics number of epochs in a deep learning algorithm
    loss_epoch = 3 + 7 * np.exp(-0.05 * epoch)
    accuracy_epoch = 0.95 - 3 * (np.log10(3 - 0.025 * epoch)) ** 2

    # Emulate the change in training loss and model accuracy as outputs while
    # learning rate changes as an input
    z = np.log10(lr)
    loss_lr = 0.6 + (np.log10(0.9 - z)) ** 2
    accuracy_lr = 0.98 - (np.log10(0.7 - 0.5 * z)) ** 2

    # Make hyperparameter tuning a multivariable optimization problem
    accuracy = k_a * accuracy_epoch * accuracy_epoch * accuracy_lr
    precision = k_p * precision_thr * accuracy_epoch * accuracy_lr
    recall = k_r * recall_thr * accuracy_epoch * accuracy_lr
    loss = k_l * loss_epoch * loss_lr
    F_1 = 2 * (precision * recall) / (precision + recall)

    # Creat a dict to retun the model results
    result = {
        "thr": thr,
        "epoch": epoch,
        "lr": lr,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "loss": loss,
        "F_1": F_1,
    }

    return result
