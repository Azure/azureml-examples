from torch.utils.data import Dataset


class AzureMLDataset(Dataset):
    def __init__(self, dataset, buffering_options=None, data_transforms=None, label_transforms=None):
        self._dataframe = dataset.to_pandas_dataframe()
        self._buffering_options = buffering_options
        self._data_transforms = data_transforms
        self._label_transforms = label_transforms

    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self, idx):
        from PIL import Image

        image_stream = self._dataframe.iloc[idx, 0]
        with image_stream.open(self._buffering_options) as f:
            image = Image.open(f)
            if self._data_transforms:
                image = self._data_transforms(image)

        label = self._dataframe.iloc[idx, 1]
        if self._label_transforms:
            label = self._label_transforms(label)

        return image, label
