from abc import abstractmethod
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.utils.common import FitRequirement


class BaseNetworkComponent(autoPyTorchSetupComponent):
    """
    Provide an abstract interface for networks
    in Auto-Pytorch
    """

    def __init__(
            self,
            random_state: Optional[np.random.RandomState] = None,
            device: Optional[torch.device] = None
    ) -> None:
        super(BaseNetworkComponent, self).__init__()
        self.network = None
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        self.add_fit_requirements([
            FitRequirement("input_shape", (numbers.Integral,), user_defined=True, dataset_property=True),
            FitRequirement("output_shape", (numbers.Integral,), user_defined=True, dataset_property=True)])

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchSetupComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """
        # Make sure that input dictionary X has the required
        # information to fit this stage
        self.check_requirements(X, y)

        input_shape = X['input_shape']
        output_shape = X['output_shape']

        self.network = self.build_network(input_shape=input_shape,
                                          output_shape=output_shape)

        # Properly set the network training device
        self.to(self.device)

        return self

    @abstractmethod
    def build_network(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> torch.nn.Module:
        """
        This method returns a pytorch network, that is dynamically built using
        a self.config that is network specific, and contains the additional
        configuration hyperparameters to build a domain specific network
        """
        raise NotImplementedError()

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        The transform function updates the network in the X dictionary.
        """
        X.update({'network': self.network})
        return X

    def get_network(self) -> nn.Module:
        """
        Return the underlying network object.
        Returns:
            model : the underlying network object
        """
        assert self.network is not None, "No network was initialized"
        return self.network

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        This common utility makes sure that the input dictionary X,
        used to fit a given component class, contains the minimum information
        to fit the given component, and it's parents
        """

        # Honor the parent requirements
        super().check_requirements(X, y)

        # For the Network, we need the number of input features,
        # to build the first network layer
        if 'input_shape' not in X.keys():
            raise ValueError("Could not find the input shape in the fit dictionary. "
                             "To fit a network, the input shape is needed to define "
                             "the hidden layers, yet the dict contains only: {}".format(X.keys()))

        # For the Network, we need the number of classes,
        # to build the last layer
        if 'output_shape' not in X:
            raise ValueError("Could not find the output shape in the fit dictionary. "
                             "To fit a network, the output shape is needed to define "
                             "the hidden layers, yet the dict contains only: {}".format(X.keys()))

    def get_network_weights(self) -> torch.nn.parameter.Parameter:
        """Returns the weights of the network"""
        assert self.network is not None, "No network was initialized"
        return self.network.parameters()

    def to(self, device: Optional[torch.device] = None) -> None:
        """Setups the network in cpu or gpu"""
        assert self.network is not None, "No network was initialized"
        if device is not None:
            self.network = self.network.to(device)
        else:
            self.network = self.network.to(self.device)

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.network.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('network', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
