from typing import List
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


def get_mnist_dataset(*args, **kwargs):
    return MNIST(*args, **kwargs)


class SingleLabelMNIST(Dataset):
    def __init__(self, dataset: Dataset, label: int):
        super().__init__()
        self.LABEL = label
        self._get_dataset(dataset)

    def _get_dataset(self, dataset: Dataset):
        self.dataset = []
        append = self.dataset.append

        for image, label in dataset:
            if label == self.LABEL:
                append(image)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.dataset[idx], self.LABEL

    def __len__(self) -> int:
        return len(self.dataset)


class MultiLabelMNIST(Dataset):
    def __init__(self, dataset: Dataset, label: List[int]):
        super().__init__()
        self.LABEL = label
        self._get_dataset(dataset)

    def _get_dataset(self, dataset: Dataset):
        self.dataset = []
        append = self.dataset.append

        for image, label in dataset:
            if label in self.LABEL:
                append((image, label))

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)
