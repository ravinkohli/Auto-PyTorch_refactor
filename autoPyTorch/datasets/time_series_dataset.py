from torch.utils import data
import numpy as np
from autoPyTorch.datasets.base_dataset import BaseDataset
from typing import Tuple, Optional, List, Sequence


class TimeSeriesRegressionDataset(BaseDataset):
    def __init__(self, target_variables: Sequence[int], sequence_length: int, n_steps: int,
                 train_tensors: Tuple[np.ndarray], val_tensors: Optional[Tuple[np.ndarray]] = None):
        _check_time_series_regression_inputs(
            target_variables=target_variables,
            sequence_length=sequence_length,
            n_steps=n_steps,
            train_tensors=train_tensors,
            val_tensors=val_tensors)
        train = _prepare_time_series_tensor(tensor=train_tensors[0], sequence_length=sequence_length, n_steps=n_steps)
        val = _prepare_time_series_tensor(tensor=val_tensors[0], sequence_length=sequence_length,
                                          n_steps=n_steps) if val_tensors is not None else None
        super().__init__(train_tensors=train, val_tensors=val)

    def create_cross_val_splits(self, cross_val_type: str, num_splits: int) -> List[Tuple[data.Dataset, data.Dataset]]:
        if self.val_tensors is not None:
            raise ValueError(f"Cannot specify validation data when using cross validation.")
        if cross_val_type == "time-series-fold":
            splits = _time_series_cross_validation(self.train_tensors[0], num_splits)
            return [(data.Subset(self, split[0]), data.Subset(self, split[1])) for split in splits]
        else:
            raise ValueError(f"Unknown cross validation type {cross_val_type}")

    def create_val_split(self, val_share: float = None) -> Tuple[data.Dataset, data.Dataset]:
        if val_share is not None:
            if self.val_tensors is not None:
                raise ValueError(
                    '`val_share` specified, but the Dataset was a given a pre-defined split at initialization already.')
            indices = np.arange(len(self))
            val_count = round(val_share * len(self))
            return data.Subset(self, indices[:val_count]), data.Subset(self, indices[val_count:])
        else:
            if self.val_tensors is None:
                raise ValueError('Please specify `val_share` or initialize with a validation dataset.')
            val_ds = BaseDataset(self.val_tensors)
            return self, val_ds


def _check_time_series_regression_inputs(target_variables: Sequence[int], sequence_length: int, n_steps: int,
                                         train_tensors: Tuple[np.ndarray],
                                         val_tensors: Optional[Tuple[np.ndarray]] = None):
    # TODO: check sequence length, target variables and n_steps
    if len(train_tensors) != 1:
        raise ValueError(f"Multiple training tensor for time series regression is not supported")
    if train_tensors[0].ndim != 3:
        raise ValueError(
            f"The training data for time series regression has to be a three-dimensional tensor of shape [P, S, V]")
    if val_tensors is not None:
        if len(val_tensors) != 1:
            raise ValueError(f"Multiple validation tensors for time series regression is not supported")
        if val_tensors[0].ndim != 3:
            raise ValueError(
                f"The validation data for time series regression has to be a three-dimensional tensor of shape [P, S, V]")


def _prepare_time_series_tensor(tensor: np.ndarray, sequence_length: int, n_steps: int) -> Tuple[
    np.ndarray, np.ndarray]:
    # TODO: implement this
    pass


def _time_series_cross_validation(tensor: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns train and validation indices respecting the temporal ordering of the data.
    Assumes the data is ordered ascending by time.
    Assumes data format is NxSxM (N records of length S with M features).
    Dummy example: [0, 1, 2, 3] with 3 folds yields
        [0] [1]
        [0, 1] [2]
        [0, 1, 2] [3]

    :param tensor: tensor holding time series data, expected to be in format NxSxM
    :param n_splits: number of cross validation splits
    :return: list of tuples of training and validation indices
    """
    assert (tensor.ndim == 3)

    split_size = int(np.ceil(tensor.shape[0] / (n_splits + 1)))

    splits = []
    train_bound = split_size
    for i in range(n_splits):
        train_indices = np.arange(min(train_bound, tensor.shape[0]))
        test_indices = np.arange(train_bound, min(
            train_bound + split_size, tensor.shape[0]))
        splits.append((train_indices, test_indices))
        train_bound += split_size
    return splits


class TimeSeriesClassificationDataset(BaseDataset):
    def __init__(self, train_tensors: Tuple[np.ndarray], val_tensors: Optional[Tuple[np.ndarray]] = None):
        super().__init__(train_tensors, val_tensors)
