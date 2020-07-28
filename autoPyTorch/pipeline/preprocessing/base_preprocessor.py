import numpy as np

import sklearn.compose
from scipy import sparse

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace

from autopytorch.pipeline.base_pipeline import BasePipeline


class DataPreprocessor(BasePipeline):
    """ This component is used to apply distinct transformations to categorical and
    numerical features of a dataset. It is built on top of sklearn's ColumnTransformer.
    """

    def __init__():
        raise NotImplementedError

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        raise NotImplementedError

    @staticmethod
    def get_properties(dataset_properties=None):
        raise NotImplementedError

    def set_hyperparameters(self, configuration, init_params=None):
        raise NotImplementedError

    def get_hyperparameter_search_space(self, dataset_properties=None):
        raise NotImplementedError

    @staticmethod
    def _get_hyperparameter_search_space_recursevely(dataset_properties, cs, transformer):
        raise NotImplementedError
