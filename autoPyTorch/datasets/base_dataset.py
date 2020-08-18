from abc import ABCMeta
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
