import numpy as np
from typing import List, Tuple


def k_fold_cross_validation(num_splits: int, indices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Standard k fold cross validation.

    :param indices: array of indices to be split
    :param num_splits: number of cross validation splits
    :return: list of tuples of training and validation indices
    """
    num_indices = len(indices)
    splits = []
    borders = list(range(0, num_indices, (num_indices + num_splits - 1) // num_splits)) + [
        num_indices]  # last split is smaller if len(self)
    for i in range(len(borders) - 1):
        lb, ub = borders[i], borders[i + 1]
        splits.append((np.concatenate([indices[:lb], indices[ub:]]),
                       indices[lb:ub]))
    return splits


def time_series_cross_validation(num_splits: int, indices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns train and validation indices respecting the temporal ordering of the data.
    Dummy example: [0, 1, 2, 3] with 3 folds yields
        [0] [1]
        [0, 1] [2]
        [0, 1, 2] [3]

    :param indices: array of indices to be split
    :param num_splits: number of cross validation splits
    :return: list of tuples of training and validation indices
    """
    split_size = int(np.ceil(len(indices) / (num_splits + 1)))

    splits = []
    train_bound = split_size
    for i in range(num_splits):
        train_indices = np.arange(min(train_bound, len(indices)))
        test_indices = np.arange(train_bound, min(
            train_bound + split_size, len(indices)))
        splits.append((train_indices, test_indices))
        train_bound += split_size
    return splits
