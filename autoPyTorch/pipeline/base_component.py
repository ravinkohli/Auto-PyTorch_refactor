from typing import Any, Dict, Optional

from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np

from sklearn.base import BaseEstimator


class AutoPytorchComponent(BaseEstimator):
    @staticmethod
    def get_properties() -> Dict[str, Any]:
        """Returns the name and other properties of the component
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space() -> ConfigurationSpace:
        raise NotImplementedError()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AutoPytorchComponent':
        """Fit the component to the given feature/target pair"""
        raise NotImplementedError()

    def set_hyperparameters(self, configuration: Configuration,
                            init_params: Optional[Dict] = None
                            ) -> 'AutoPytorchComponent':
        """
        Applies a configuration to the given component.
        This method translate a hierarchical configuration key,
        to an actual parameter of the autoPyTorch component.

        Args:
            configuration (Configuration): which configuration to apply to
                the chosen component
            init_params (Optional[Dict[str, any]]): Optional arguments to
                initialize the chosen component

        Returns:
            self: returns an instance of self
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        """A representation of self"""
        raise NotImplementedError()
