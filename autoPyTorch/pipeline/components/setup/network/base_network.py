import numbers
from typing import Any, Dict

import torch

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class BaseNetworkComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for networks
    in Auto-Pytorch"""

    def __init__(self) -> None:
        self.network = None

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'network': self.network})
        return X

    def get_network(self) -> torch.nn.Module:
        """Return the underlying network object.
        Returns:
            model : the underlying network object
        """
        assert self.network is not None, "No network was fit"
        return self.network

    def check_requirements(self, X: Dict[str, Any]) -> None:
        """ This common utility makes sure that the input dictionary X,
        used to fit a given component class, contains the minimum information
        to fit the given component, and it's parents
        """

        # Honor the parent requirements
        super().check_requirements(X)

        # For the Network, we need the number of input features,
        # to build the first network layer
        if 'num_features' not in X.keys():
            print(f"wjkajdkasjkdlas {X.keys()}")
            print(f" WHAT {'num_features' not in X or not isinstance(X['num_features'], int)}")
            print(f" THE {'num_features' not in X}")
            print(f"{not isinstance(X['num_features'], int)}")
            raise ValueError("Could not parse the number of input features in the fit dictionary "
                             "To fit a network, the number of features is needed to define "
                             "the hidden layers, yet the dict contains only: {}".format(
                                 X.keys()
                             )
                             )

        assert isinstance(X['num_features'], numbers.Integral), "num_features: {}".format(
            type(X['num_features'])
        )

        # For the Network, we need the number of classes,
        # to build the last layer
        if 'num_classes' not in X:
            raise ValueError("Could not parse the number of classes in the fit dictionary "
                             "To fit a network, the number of classes is needed to define "
                             "the hidden layers, yet the dict contains only: {}".format(
                                 X.keys()
                             )
                             )
        assert isinstance(X['num_classes'], numbers.Integral), "num_classes: {}".format(
            type(X['num_classes'])
        )

    @classmethod
    def get_activations_dict(cls) -> Dict[str, torch.nn.Module]:
        """
        This method highlights the activations that can be used,
        when dynamically building a network.
        """
        return {
            'relu': torch.nn.ReLU,
            'sigmoid': torch.nn.Sigmoid,
            'tanh': torch.nn.Tanh,
            'leakyrelu': torch.nn.LeakyReLU,
            'selu': torch.nn.SELU,
            'rrelu': torch.nn.RReLU,
            'tanhshrink': torch.nn.Tanhshrink,
            'hardtanh': torch.nn.Hardtanh,
            'elu': torch.nn.ELU,
            'prelu': torch.nn.PReLU,
        }

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.network.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('network', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
