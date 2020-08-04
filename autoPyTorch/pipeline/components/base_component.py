import importlib
import inspect
import pkgutil
import sys
from collections import OrderedDict
from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

from sklearn.base import BaseEstimator


def find_components(
    package: str,
    directory: str,
    base_class: BaseEstimator
) -> Dict[str, BaseEstimator]:
    """Utility to find component on a given directory,
    that inherit from base_class
    Args:
        package (str): The associated package that contains the components
        directory (str): The directory from which to extract the components
        base_class (BaseEstimator): base class to filter out desired components
            that don't inherit from this class
    """
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    # TODO test if the obj implements the interface
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[module_name] = classifier

    return components


class autoPyTorchComponent(BaseEstimator):
    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        """Get the properties of the underlying algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]): Describes the dataset
               to work on
        Returns:
            Dict[str, Any]: Properties of the algorithm
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]): Describes the dataset
               to work on

        Returns:
            ConfigurationSpace: The configuration space of this algorithm.
        """
        raise NotImplementedError()

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> BaseEstimator:
        """The fit function calls the fit function of the underlying
        model and returns `self`.

        Args:
            X (np.ndarray): Training data
            y (np.ndarray): target data

        Returns:
            self : returns an instance of self.
        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def set_hyperparameters(self,
                            configuration: Configuration,
                            init_params: Optional[Dict[str, Any]] = None
                            ) -> BaseEstimator:
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
            An instance of self
        """
        params = configuration.get_dictionary()

        for param, value in params.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' %
                                 (param, str(self)))
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError('Cannot set init param %s for %s because '
                                     'the init param does not exist.' %
                                     (param, str(self)))
                setattr(self, param, value)

        return self

    def __str__(self) -> str:
        """Representation of the current Component"""
        name = self.get_properties()['name']
        return "autoPyTorch.pipeline %s" % name
