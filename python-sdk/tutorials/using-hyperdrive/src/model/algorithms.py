"""Script that runs a fake training function"""


def train_fake_model(initial_lr: float) -> dict:

    """
    Parameters
    ----------
    initial_lr: float
        Initial learning rate used to simulate training

    Returns
    -------
    result: dict
        List of metrics calculated using learning rate
    """

    multiplier = 1 / initial_lr

    accuracy = (multiplier * 0.99) / 100
    precision = (multiplier * 0.85) / 100
    recall = (multiplier * 0.9) / 100

    result = {
              "initial_lr": initial_lr,
              "accuracy": accuracy,
              "precision": precision,
              "recall": recall
             }

    return result
