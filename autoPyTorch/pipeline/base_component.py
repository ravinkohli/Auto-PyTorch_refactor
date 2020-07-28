from abc import ABCMeta

import numpy as np
from ConfigSpace import Configuration
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state

from sklearn.base import BaseEstimator


class AutoPytorchComponent(BaseEstimator):
    @staticmethod
    def get_properties():
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space():
        raise NotImplementedError()

    def fit(self, X, y):
        raise NotImplementedError()

    def set_hyperparameters(self, configuration, init_params=None):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

