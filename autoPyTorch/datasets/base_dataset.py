from abc import ABC
from torch.utils.data import Dataset, DataLoader,Subset

class BaseDataset(ABC):
    def __init__(self, X_train, Y_train, X_valid=None, Y_valid=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        assert self.X_valid is None == self.Y_valid is None, "Either both X/Y valid should be provided or None."
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def create_cross_val_splits(self, cross_val_type, num_splits):
        return [(Subset,Subset)]
    def create_val_split(self, val_share=None):
        # none nur für valid 
        return (Subset,Subset)

class TabularDataset(BaseDataset):
    def __init__(self, *args):
        super().__init__(*args)
        # entscheide was für felder wir haben.


Input: Numpy array(s) train/val x/y