from abc import ABCMeta
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from torch.utils.data import Dataset, Subset

import torchvision

from autoPyTorch.datasets.resampling_strategy import (
    CROSS_VAL_FN,
    CrossValTypes,
    DEFAULT_RESAMPLING_PARAMETERS,
    HOLDOUT_FN,
    HoldoutValTypes,
    is_stratified,
)

BASE_DATASET_INPUT = Union[Tuple[np.ndarray, np.ndarray], Dataset]


def check_valid_data(data: Any) -> None:
    if not (hasattr(data, '__getitem__') and hasattr(data, '__len__')):
        raise ValueError(
            'The specified Data for Dataset does either not have a __getitem__ or a __len__ attribute.')


def type_check(train_tensors: BASE_DATASET_INPUT, val_tensors: Optional[BASE_DATASET_INPUT] = None) -> None:
    for i in range(len(train_tensors)):
        check_valid_data(train_tensors[i])
    if val_tensors is not None:
        for i in range(len(val_tensors)):
            check_valid_data(val_tensors[i])


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        train_tensors: BASE_DATASET_INPUT,
        val_tensors: Optional[BASE_DATASET_INPUT] = None,
        test_tensors: Optional[BASE_DATASET_INPUT] = None,
        resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.holdout_validation,
        resampling_strategy_args: Optional[Dict[str, Any]] = None,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 42,
        transforms: Optional[torchvision.transforms.Compose] = None
    ):
        """
        :param train_tensors: A tuple of objects that have a __len__ and a __getitem__ attribute.
        :param val_tensors: A optional tuple of objects that have a __len__ and a __getitem__ attribute.
        :param shuffle: Whether to shuffle the data before performing splits
        """
        if not hasattr(train_tensors[0], 'shape'):
            type_check(train_tensors, val_tensors)
        self.train_tensors = train_tensors
        self.val_tensors = val_tensors
        self.test_tensors = test_tensors
        self.cross_validators: Dict[str, CROSS_VAL_FN] = {}
        self.holdout_validators: Dict[str, HOLDOUT_FN] = {}
        self.rand = np.random.RandomState(seed=seed)
        self.shuffle = shuffle
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args

        # Make sure cross validation splits are created once
        self.splits = None  # type: Optional[List]

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.transform = transforms

    def update_transform(self, transform: Optional[torchvision.transforms.Compose]
                         ) -> 'BaseDataset':
        """
        During the pipeline execution, the pipeline object might propose transformations
        as a product of the current pipeline configuration being tested.

        This utility allows to return a self with the updated transformation, so that
        a dataloader can yield this dataset with the desired transformations

        Args:
            transform (torchvision.transforms.Compose): The transformations proposed
                by the current pipeline

        Returns:
            self: A copy of the update pipeline
        """
        self.transform = transform
        return self

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:

        if hasattr(self.train_tensors[0], 'loc'):
            X = self.train_tensors[0].iloc[index].to_numpy()
        else:
            X = self.train_tensors[0][index]
        if self.transform:
            X = self.transform(X)

        # In case of prediction, the targets are not provided
        Y = self.train_tensors[1]
        if Y is not None:
            Y = Y[index]
        else:
            Y = None

        return X, Y

    def __len__(self) -> int:
        return self.train_tensors[0].shape[0]

    def _get_indices(self) -> np.ndarray:
        if self.shuffle:
            indices = self.rand.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices

    def create_cross_val_splits(self,
                                cross_val_type: CrossValTypes,
                                num_splits: int) -> None:
        """
        This function creates the cross validation split for the given task.

        It is done once per dataset to have comparable results among pipelines
        """
        # Create just the split once
        if self.splits is None:
            if not isinstance(cross_val_type, CrossValTypes):
                raise NotImplementedError(f'The selected `cross_val_type` "{cross_val_type}" is not implemented.')
            kwargs = {}
            if is_stratified(cross_val_type):
                # we need additional information about the data for stratification
                kwargs["stratify"] = self.train_tensors[-1]
            self.splits = self.cross_validators[cross_val_type.name](
                num_splits, self._get_indices(), **kwargs)
        return

    def create_val_split(self,
                         holdout_val_type: Optional[HoldoutValTypes] = None,
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
            if not isinstance(holdout_val_type, HoldoutValTypes):
                raise NotImplementedError(f'The specified `holdout_val_type` "{holdout_val_type}" is not supported.')
            kwargs = {}
            if is_stratified(holdout_val_type):
                # we need additional information about the data for stratification
                kwargs["stratify"] = self.train_tensors[-1]
            train, val = self.holdout_validators[holdout_val_type.name](val_share, self._get_indices(), **kwargs)
            return Subset(self, train), Subset(self, val)
        else:
            if self.val_tensors is None:
                raise ValueError('Please specify `val_share` or initialize with a validation dataset.')
            val = BaseDataset(self.val_tensors)
            return self, val

    def get_dataset_for_training(self, split: int) -> Tuple[Dataset, Dataset]:
        """
        The above split methods employ the Subset to internally subsample the whole dataset.

        During training, we need access to one of those splits. This is a handy function
        to provide training data to fit a pipeline

        Args:
            split (int): The desired subset of the dataset to split and use

        Returns:
            Dataset: the reduced dataset to be used for testing
        """
        if isinstance(self.resampling_strategy, HoldoutValTypes):
            # Regardless of the split, there is a single dataset
            val_share = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'val_share', None)
            if self.resampling_strategy_args is not None:
                val_share = self.resampling_strategy_args.get('val_share', val_share)
            return self.create_val_split(
                holdout_val_type=self.resampling_strategy,
                val_share=val_share,
            )
        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'num_splits', None),
            if self.resampling_strategy_args is not None:
                num_splits = self.resampling_strategy_args.get('num_splits', num_splits)
            # Create the split if it was not created before
            self.create_cross_val_splits(
                cross_val_type=self.resampling_strategy,
                num_splits=cast(int, num_splits),
            )

            # Subset creates a dataset
            # Asser for mypy -- self.splits is created above in self.create_cross_val_splits
            assert self.splits is not None
            return (Subset(self, self.splits[split][0]), Subset(self, self.splits[split][1]))
        else:
            raise ValueError(f"Unsupported resampling strategy {self.resampling_strategy}")

    def replace_data(self, X_train: BASE_DATASET_INPUT, X_test: Optional[BASE_DATASET_INPUT]) -> 'BaseDataset':
        """
        To speed up the training of small dataset, early pre-processing of the data
        can be made on the fly by the pipeline.

        In this case, we replace the original train/test tensors by this pre-processed version

        Args:
            X_train (np.ndarray): the pre-processed (imputation/encoding/...) train data
            X_test (np.ndarray): the pre-processed (imputation/encoding/...) test data

        Returns:
            self
        """
        self.train_tensors = (X_train, self.train_tensors[1])
        if X_test is not None and self.test_tensors is not None:
            self.test_tensors = (X_test, self.test_tensors[1])
        return self
