from torch.utils import data
import numpy as np
from autoPyTorch.datasets.base_dataset import BaseDataset
from typing import Tuple, Optional, Set
from .cross_validation import time_series_cross_validation, k_fold_cross_validation


class TimeSeriesForecastingDataset(BaseDataset):
    def __init__(self, target_variables: Set[int], sequence_length: int, n_steps: int,
                 train_tensors: Tuple[np.ndarray, ...], val_tensors: Optional[Tuple[np.ndarray, ...]] = None):
        """

        :param target_variables: The indices of the variables you want to forecast
        :param sequence_length: The amount of past data you want to use to forecast future value
        :param n_steps: The number of steps you want to forecast into the future
        :param train_tensors: Tuple with one tensor holding the training data
        :param val_tensors: Tuple with one tensor holding the validation data
        """
        _check_time_series_forecasting_inputs(
            target_variables=target_variables,
            sequence_length=sequence_length,
            n_steps=n_steps,
            train_tensors=train_tensors,
            val_tensors=val_tensors)
        train = _prepare_time_series_forecasting_tensor(tensor=train_tensors[0],
                                                        target_variables=target_variables,
                                                        sequence_length=sequence_length,
                                                        n_steps=n_steps)
        val = _prepare_time_series_forecasting_tensor(tensor=val_tensors[0],
                                                      target_variables=target_variables,
                                                      sequence_length=sequence_length,
                                                      n_steps=n_steps) if val_tensors is not None else None
        self.cross_validators["time_series_cross_validation"] = time_series_cross_validation
        super().__init__(train_tensors=train, val_tensors=val)

    def _get_data_indices(self) -> np.ndarray:
        return np.arange(len(self))


def _check_time_series_forecasting_inputs(target_variables: Set[int], sequence_length: int, n_steps: int,
                                          train_tensors: Tuple[np.ndarray, ...],
                                          val_tensors: Optional[Tuple[np.ndarray, ...]] = None):
    if len(train_tensors) != 1:
        raise ValueError(f"Multiple training tensors for time series regression is not supported.")
    if train_tensors[0].ndim != 3:
        raise ValueError(
            f"The training data for time series regression has to be a three-dimensional tensor of shape PxLxM.")
    if val_tensors is not None:
        if len(val_tensors) != 1:
            raise ValueError(f"Multiple validation tensors for time series regression is not supported.")
        if val_tensors[0].ndim != 3:
            raise ValueError(
                f"The validation data for time series regression "
                f"has to be a three-dimensional tensor of shape PxLxM.")
    _, time_series_length, num_features = train_tensors[0].shape
    if sequence_length + n_steps > time_series_length:
        raise ValueError(f"Invalid sequence length: Cannot create dataset "
                         f"using sequence_length={sequence_length} and n_steps={n_steps} "
                         f"when the time series are of length {time_series_length}")
    for t in target_variables:
        if t < 0 or t >= num_features:
            raise ValueError(f"Target variable {t} is out of bounds. Number of features is {num_features}, "
                             f"so each target variable has to be between 0 and {num_features - 1}.")


def _prepare_time_series_forecasting_tensor(tensor: np.ndarray,
                                            target_variables: Set[int],
                                            sequence_length: int,
                                            n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    population_size, time_series_length, num_features = tensor.shape
    num_targets = len(target_variables)
    num_datapoints = time_series_length - sequence_length - n_steps + 1
    x_tensor = np.zeros((num_datapoints, population_size, sequence_length, num_features), dtype=np.float)
    y_tensor = np.zeros((num_datapoints, population_size, num_targets), dtype=np.float)

    for p in range(population_size):
        for i in range(num_datapoints):
            x_tensor[i, p, :, :] = tensor[p, i:i + sequence_length, :]
            y_tensor[i, p, :] = tensor[p, i + sequence_length + n_steps - 1, target_variables]

    # get rid of population dimension by reshaping
    x_tensor = x_tensor.reshape((-1, sequence_length, num_features))
    y_tensor = y_tensor.reshape((-1, num_targets))
    return x_tensor, y_tensor


class TimeSeriesClassificationDataset(BaseDataset):
    def __init__(self, train_tensors: Tuple[np.ndarray, ...], val_tensors: Optional[Tuple[np.ndarray, ...]] = None):
        _check_time_series_classification_inputs(train_tensors=train_tensors, val_tensors=val_tensors)
        self.cross_validators["k_fold_cross_validation"] = k_fold_cross_validation
        super().__init__(train_tensors, val_tensors)

    def _get_data_indices(self) -> np.ndarray:
        return np.random.permutation(len(self))


def _check_time_series_classification_inputs(train_tensors: Tuple[np.ndarray, ...],
                                             val_tensors: Optional[Tuple[np.ndarray, ...]]):
    if len(train_tensors) != 2:
        raise ValueError(f"There must be exactly two training tensors for time series classification. "
                         f"The first one containing the data and the second one containing the class labels.")
    if train_tensors[0].ndim != 3:
        raise ValueError(
            f"The training data for time series classification has to be a three-dimensional tensor of shape NxSxM.")
    if train_tensors[1].ndim != 2:
        raise ValueError(
            f"The training targets for time series classification have to be of shape NxC."
        )
    if val_tensors is not None:
        if len(val_tensors) != 2:
            raise ValueError(f"There must be exactly two validation tensors for time series classification. "
                             f"The first one containing the data and the second one containing the class labels.")
        if val_tensors[0].ndim != 3:
            raise ValueError(
                f"The validation data for time series classification has to be a "
                f"three-dimensional tensor of shape NxSxM.")
        if val_tensors[1].ndim != 2:
            raise ValueError(
                f"The training targets for time series classification have to be of shape NxC."
            )
