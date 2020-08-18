from abc import ABCMeta
from torch.utils.data import Dataset, Subset
import numpy as np
from typing import Optional, Tuple, List, Any, Dict
from autoPyTorch.datasets.cross_validation import CROSS_VAL_FN, HOLDOUT_FN, is_stratified

BASE_DATASET_INPUT = Tuple[Any, ...]


def check_valid_data(data: Any) -> None:
    if not (hasattr(data, '__getitem__') and hasattr(data, '__len__')):
        raise ValueError(
            'The specified Data for Dataset does either not have a __getitem__ or a __len__ attribute.')


def type_check(train_tensors: BASE_DATASET_INPUT, val_tensors: Optional[BASE_DATASET_INPUT] = None) -> None:
    for t in train_tensors:
        check_valid_data(t)
    if val_tensors is not None:
        for t in val_tensors:
            check_valid_data(t)


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 train_tensors: BASE_DATASET_INPUT,
                 val_tensors: Optional[BASE_DATASET_INPUT] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42):
        """
        :param train_tensors: A tuple of objects that have a __len__ and a __getitem__ attribute.
        :param val_tensors: A optional tuple of objects that have a __len__ and a __getitem__ attribute.
        :param shuffle: Whether to shuffle the data before performing splits
        """
        type_check(train_tensors, val_tensors)
        self.train_tensors = train_tensors
        self.val_tensors = val_tensors
        self.cross_validators: Dict[str, CROSS_VAL_FN] = {}
        self.holdout_validators: Dict[str, HOLDOUT_FN] = {}
        self.rand = np.random.RandomState(seed=seed)
        self.shuffle = shuffle

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        return tuple(t[index] for t in self.train_tensors)

    def __len__(self) -> int:
        return len(self.train_tensors[0])

    def _get_indices(self) -> np.ndarray:
        if self.shuffle:
            indices = self.rand.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices

    def create_cross_val_splits(self,
                                cross_val_type: str,
                                num_splits: int) -> List[Tuple[Dataset, Dataset]]:
        if cross_val_type not in self.cross_validators:
            raise NotImplementedError(f'The selected `cross_val_type` "{cross_val_type}" is not implemented.')
        kwargs = {}
        if is_stratified(cross_val_type):
            # we need additional information about the data for stratification
            kwargs["stratify"] = self.train_tensors[-1]
        splits = self.cross_validators[cross_val_type](num_splits, self._get_indices(), **kwargs)
        return [(Subset(self, split[0]), Subset(self, split[1])) for split in splits]

    def create_val_split(self,
                         holdout_val_type: Optional[str] = None,
                         val_share: Optional[float] = None) -> Tuple[Dataset, Dataset]:
        if val_share is not None:
            if holdout_val_type is None:
                raise ValueError(
                    '`val_share` specified, but `holdout_val_type` not specified.'
                )
            if self.val_tensors is not None:
                raise ValueError(
                    '`val_share` specified, but the Dataset was a given a pre-defined split at initialization already.')
            if val_share < 0 or val_share > 1:
                raise ValueError(f"`val_share` must be between 0 and 1, got {val_share}.")
            if holdout_val_type not in self.cross_validators:
                raise NotImplementedError(f'The specified `holdout_val_type` "{holdout_val_type}" is not supported.')
            kwargs = {}
            if is_stratified(holdout_val_type):
                # we need additional information about the data for stratification
                kwargs["stratify"] = self.train_tensors[-1]
            train, val = self.holdout_validators[holdout_val_type](val_share, self._get_indices(), **kwargs)
            return Subset(self, train), Subset(self, val)
        else:
            if self.val_tensors is None:
                raise ValueError('Please specify `val_share` or initialize with a validation dataset.')
            val = BaseDataset(self.val_tensors)
            return self, val
=======
from torch.utils import data
import numpy as np
from typing import Optional

class BaseDataset(data.Dataset,metaclass=ABCMeta):
    def __init__(self, X_train: np.array, Y_train: np.array, X_valid: Optional[np.array]=None, Y_valid: Optional[np.array]=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        if self.X_valid is None != self.Y_valid is None:
            raise ValueError("Either both X and Y valid sets should be provided or None.")

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def create_cross_val_splits(self, cross_val_type, num_splits):
        indices = np.random.permutation(len(self))
        if cross_val_type == 'k-fold-crossvalidation':
            splits = []
            borders = list(range(0,len(self),(len(self)+num_splits-1)//num_splits)) + [len(self)] # last split is smaller if len(self)
            for i in range(len(borders)-1):
                lb,ub = borders[i],borders[i+1]
                splits.append((data.Subset(self,np.concatenate((indices[:lb],indices[ub:]))),data.Subset(self,indices[lb:ub])))
        else:
            NotImplementedError(f'The selected `cross_val_type` "{cross_val_type}" is not implemented.')
        return splits

    def create_val_split(self, val_share=None):
        indices = np.random.permutation(len(self))
        val_count = round(val_share*len(self))
        if val_share is None:
            if self.X_valid is not None:
                raise ValueError('`val_share` specified, but the Dataset was a given a pre-defined split at initializiation already.')
            return (data.Subset(self,indices[:val_count]),data.Subset(self,indices[val_count:]))
        else:
            if self.X_valid is None:
                raise ValueError('Please specify `val_share` or initialize with a validation dataset.')
            val_ds = BaseDataset(self.X_valid,self.Y_valid)
            return (self,val_ds)
