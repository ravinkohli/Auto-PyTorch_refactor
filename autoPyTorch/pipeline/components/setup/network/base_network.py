import numbers
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np

import torch

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.utils.common import FitRequirement


class BaseNetworkComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for networks
    in Auto-Pytorch"""

    def __init__(
        self,
        intermediate_activation: str,
        final_activation: Optional[str],
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        self.network = None
        self.intermediate_activation = intermediate_activation
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._fit_requirements = [FitRequirement('num_features', numbers.Integral),
                                  FitRequirement('num_classes', numbers.Integral)]

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

        in_features = X['num_features']
        out_features = X['num_classes']

        self.network = self.build_network(in_features, out_features)

        # Properly set the network training device
        self.to(self.device)

        return self

    @abstractmethod
    def build_network(self, in_feature: int, out_features: int) -> torch.nn.Module:
        """This method returns a pytorch network, that is dynamically built
        using:

            common network arguments from the base class:
                * intermediate_activation
                * final_activation

            a self.config that is network specific, and contains the additional
            configuration hyperparameters to build a domain specific network
        """
        raise NotImplementedError()

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
        assert self.network is not None, "No network was initialized"
        return self.network

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

    def predict(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Performs batched prediction given a loader object
        """
        assert self.network is not None
        self.network.eval()

        # Batch prediction
        Y_batch_preds = list()

        for i, (X_batch, Y_batch) in enumerate(loader):
            # Predict on batch
            X_batch = torch.autograd.Variable(X_batch).to(self.device)

            Y_batch_pred = self.network(X_batch).detach().cpu()
            Y_batch_preds.append(Y_batch_pred)

        return torch.cat(Y_batch_preds, 0)

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.network.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('network', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
