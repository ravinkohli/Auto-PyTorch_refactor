import numpy as np
from torch.utils.data import Dataset, TensorDataset
import torch
from PIL import Image
from autoPyTorch.datasets.base_dataset import BaseDataset
from typing import Tuple, Optional, Union, List
from autoPyTorch.datasets.cross_validation import CrossValTypes, HoldoutValTypes, get_cross_validators, \
    get_holdout_validators

IMAGE_DATASET_INPUT = Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]


class ImageDataset(BaseDataset):
    def __init__(self,
                 train: IMAGE_DATASET_INPUT,
                 val: Optional[IMAGE_DATASET_INPUT] = None):
        _check_image_inputs(train=train, val=val)
        train = _create_image_dataset(data=train)
        if val is not None:
            val = _create_image_dataset(data=val)
        super().__init__(train_tensors=(train,), val_tensors=(val,), shuffle=True)
        self.cross_validators = get_cross_validators(
            CrossValTypes.stratified_k_fold_cross_validation,
            CrossValTypes.k_fold_cross_validation,
            CrossValTypes.shuffle_split_cross_validation,
            CrossValTypes.stratified_shuffle_split_cross_validation
        )
        self.holdout_validators = get_holdout_validators(
            HoldoutValTypes.train_val_split,
            HoldoutValTypes.stratified_train_val_split
        )


def _check_image_inputs(train: IMAGE_DATASET_INPUT,
                        val: Optional[IMAGE_DATASET_INPUT] = None):
    if not isinstance(train, Dataset):
        if len(train[0]) != len(train[1]):
            raise ValueError(
                f"expected train inputs to have the same length, but got lengths {len(train[0])} and {len(train[1])}")
        if val is not None:
            if len(val[0]) != len(val[1]):
                raise ValueError(
                    f"expected val inputs to have the same length, but got lengths {len(train[0])} and {len(train[1])}")


def _create_image_dataset(data: IMAGE_DATASET_INPUT) -> Dataset:
    # if user already provided a dataset, use it
    if isinstance(data, Dataset):
        return data
    # if user provided list of file paths, create a file path dataset
    if isinstance(data[0], list):
        return _FilePathDataset(file_paths=data[0], targets=data[1])
    # if user provided the images as numpy tensors use them directly
    else:
        return TensorDataset(torch.tensor(data[0]), torch.tensor(data[1]))


class _FilePathDataset(Dataset):
    def __init__(self, file_paths: List[str], targets: np.ndarray):
        self.file_paths = file_paths
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Image, torch.Tensor]:
        with open(self.file_paths[index], "rb") as f:
            img = Image.open(f).convert("RGB")
        return img, torch.tensor(self.targets[index])

    def __len__(self) -> int:
        return len(self.file_paths)
