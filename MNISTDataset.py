from typing import List, Dict, Union
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST


DATASET_PARAM = {"root": "./data", "train": True, "download": True, "transform": None}

DATALOADER_PARAM = {
    "batch_size": 128,
    "shuffle": True,
    "num_workers": 1,
    "pin_memory": True,
}


class SingleLabelMNIST(Dataset):
    def __init__(self, dataset: Dataset, label: int):
        super().__init__()
        self.LABEL = label
        self._get_dataset(dataset)

    def _get_dataset(self, dataset: Dataset):
        d = []
        append = d.append

        for image, label in dataset:
            if label == self.LABEL:
                append(image)

        self.dataset = tuple(d)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.dataset[idx], self.LABEL

    def __len__(self) -> int:
        return len(self.dataset)


class MultiLabelMNIST(Dataset):
    def __init__(self, dataset: Dataset, labels: List[int]):
        super().__init__()
        self.LABEL = labels
        self._get_dataset(dataset)

    def _get_dataset(self, dataset: Dataset):
        d = []
        append = d.append

        for image, label in dataset:
            if label in self.LABEL:
                append((image, label))

        self.dataset = tuple(d)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


def get_mnist_dataset(dataset_param: Dict = DATASET_PARAM) -> Dataset:
    return MNIST(**dataset_param)


def get_mnist_dataloader(
    label: Union[int, List[int]],
    dataset_param: Dict = DATASET_PARAM,
    dataloader_param: Dict = DATALOADER_PARAM,
) -> DataLoader:
    d = get_mnist_dataset(dataset_param)
    d = (
        SingleLabelMNIST(d, label)
        if isinstance(label, int)
        else MultiLabelMNIST(d, label)
    )
    return DataLoader(d, **dataloader_param)
