import typing

import numpy as np

import torch

import torchvision


class CustomXYTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support for transformations

    Attributes:

    Arguments:

    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 transform: typing.Optional[torchvision.transforms.Compose] = None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index]

        if self.transform:
            X = self.transform(X)

        if self.y is not None:
            y = self.y[index]
        else:
            y = None

        return X, y

    def __len__(self) -> int:
        return self.X.shape[0]
