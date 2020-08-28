import numpy as np
from torch.utils.data import Dataset
from autoPyTorch.datasets.base_dataset import BaseDataset
from typing import Tuple, Optional, Union, List
from autoPyTorch.datasets.cross_validation import k_fold_cross_validation, \
    holdout_validation, \
    stratified_holdout_validation


class ImageDataset(BaseDataset):
    def __init__(self,
                 train: Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]],
                 val: Optional[Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]] = None):
        _type_check(train, "train")
        _check_image_inputs(train=train, val=val)
        super().__init__(train_tensors=train, val_tensors=val, shuffle=True)
        self.cross_validators.update(
            {"k_fold_cross_validation": k_fold_cross_validation}
        )
        self.holdout_validators.update(
            {"holdout_validation": holdout_validation,
             "stratified_holdout_validation": stratified_holdout_validation}
        )


def _type_check(t: Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]], name: str):
    # TODO: finish this
    if isinstance(t, Dataset):
        raise ValueError("")
    elif isinstance(t, tuple):
        if len(t) != 2:
            raise ValueError("")
        pass
    else:
        raise TypeError(f"expected input `{name}` to be of type torch.utils.data.Dataset, {type(Dataset)}")


def _check_image_inputs(train: Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]],
                        val: Optional[Union[Dataset, Tuple[Union[np.ndarray, List[str]], np.ndarray]]]):
    # TODO: finish this
    _type_check(train, "train")
    if val is not None:
        _type_check(val, "val")
