from abc import ABCMeta
from torch.utils import data
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable, Any


def check_valid_data(data: Any) -> None:
    if not (hasattr(data, '__getitem__') and hasattr(data, '__len__')):
        raise ValueError(
            'The specified Data for Dataset does either not have a __getitem__ or a __len__ attribute.')


def type_check(train_tensors: Tuple[Any, ...], val_tensors: Optional[Tuple[Any, ...]]) -> None:
    for t in train_tensors:
        check_valid_data(t)
    if val_tensors is not None:
        for t in val_tensors:
            check_valid_data(t)


class BaseDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self, train_tensors: Tuple[Any, ...], val_tensors: Optional[Tuple[Any, ...]] = None):
        """
        :param train_tensors: A tuple of objects that have a __len__ and a __getitem__ attribute.
        :param val_tensors: A optional tuple of objects that have a __len__ and a __getitem__ attribute.
        """
        type_check(train_tensors, val_tensors)
        self.train_tensors = train_tensors
        self.val_tensors = val_tensors
        self.cross_validators = {}  # type: Dict[str, Callable[[int, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        return tuple(t[index] for t in self.train_tensors)

    def __len__(self) -> int:
        return self.train_tensors[0].shape[0]

    def _get_data_indices(self) -> np.ndarray:
        raise NotImplementedError

    def create_cross_val_splits(self, cross_val_type: str, num_splits: int) -> List[Tuple[data.Dataset, data.Dataset]]:
        indices = self._get_data_indices()
        if cross_val_type not in self.cross_validators:
            raise NotImplementedError(f'The selected `cross_val_type` "{cross_val_type}" is not implemented.')
        splits = self.cross_validators[cross_val_type](num_splits, indices)
        return [(data.Subset(self, split[0]), data.Subset(self, split[1])) for split in splits]

    def create_val_split(self, val_share: float = None) -> Tuple[data.Dataset, data.Dataset]:
        if val_share is not None:
            if self.val_tensors is not None:
                raise ValueError(
                    '`val_share` specified, but the Dataset was a given a pre-defined split at initialization already.')
            indices = self._get_data_indices()
            val_count = round(val_share * len(self))
            return data.Subset(self, indices[:val_count]), data.Subset(self, indices[val_count:])
        else:
            if self.val_tensors is None:
                raise ValueError('Please specify `val_share` or initialize with a validation dataset.')
            val_ds = BaseDataset(self.val_tensors)
            return self, val_ds
