from abc import ABCMeta
from torch.utils import data
import numpy as np
from typing import Optional, Tuple, Mapping, Any

class BaseDataset(data.Dataset,metaclass=ABCMeta):
    def __init__(self, train_tensors: Tuple[Mapping[int,Any]], val_tensors: Optional[Tuple[Mapping[int,Any]]]=None):
        self.train_tensors = train_tensors
        self.val_tensors = val_tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.train_tensors)

    def __len__(self):
        return len(self.train_tensors)

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
            if self.val_tensors is not None:
                raise ValueError('`val_share` specified, but the Dataset was a given a pre-defined split at initializiation already.')
            return (data.Subset(self,indices[:val_count]),data.Subset(self,indices[val_count:]))
        else:
            if self.val_tensors is None:
                raise ValueError('Please specify `val_share` or initialize with a validation dataset.')
            val_ds = BaseDataset(self.val_tensors)
            return (self,val_ds)
