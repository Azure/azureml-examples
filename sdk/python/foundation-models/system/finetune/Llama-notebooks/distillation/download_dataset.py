from datasets import load_dataset
from abc import ABC


class InputDataset(ABC):
    def __init__(self):
        super().__init__()
        (
            self.train_data_file_name,
            self.test_data_file_name,
            self.eval_data_file_name,
        ) = (None, None, None)


class CQnAHuggingFaceInputDataset(InputDataset):
    """
    Loads the HuggingFace dataset
    """

    def __init__(self):
        super().__init__()

    def load_hf_dataset(
        self,
        dataset_name,
        train_sample_size=10,
        val_sample_size=10,
        test_sample_size=10,
        train_split_name="train",
        val_split_name="validation",
        test_split_name="test",
    ):
        full_dataset = load_dataset(dataset_name)

        if val_split_name is not None:
            train_data = full_dataset[train_split_name].select(range(train_sample_size))
            val_data = full_dataset[val_split_name].select(range(val_sample_size))
            test_data = full_dataset[test_split_name].select(range(test_sample_size))
        else:
            train_val_data = full_dataset[train_split_name].select(
                range(train_sample_size + val_sample_size)
            )
            train_data = train_val_data.select(range(train_sample_size))
            val_data = train_val_data.select(
                range(train_sample_size, train_sample_size + val_sample_size)
            )
            test_data = full_dataset[test_split_name].select(range(test_sample_size))

        return train_data, val_data, test_data
