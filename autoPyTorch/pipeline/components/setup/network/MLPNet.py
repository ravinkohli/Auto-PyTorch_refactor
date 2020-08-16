from typing import Any, Dict, List, Optional

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

import torch

from autoPyTorch.pipeline.components.setup.network.base_network import BaseNetworkComponent


class MLPNet(BaseNetworkComponent):
    """
    This component automatically creates a Multi Layer Perceptron based on a given config.

    This MLP allows for:
        - Different number of layers
        - Specifying the activation. But this activation is shared among layers
        - Using or not dropout
        - Specifying the number of units per layers

    Args:
        T_0 (int): Number of iterations for the first restart
        T_mult (int):  A factor increases T_{i} after a restart
        random_state (Optional[np.random.RandomState]): random state
    """

    def __init__(
        self,
        num_layers: int,
        activation: str,
        use_dropout: bool,
        random_state: Optional[np.random.RandomState] = None,
        **kwargs: Any
    ):

        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.random_state = random_state
        self.use_dropout = use_dropout
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseNetworkComponent:
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
        self.check_requirements(X)

        layers = list()  # type: List[torch.nn.Module]
        in_features = X['num_features']
        out_features = X['num_classes']

        self._add_layer(layers, in_features, self.config['num_units_1'], 1)

        for i in range(2, self.num_layers + 1):
            self._add_layer(layers, self.config["num_units_%d" % (i - 1)],
                            self.config["num_units_%d" % i], i)

        layers.append(torch.nn.Linear(self.config["num_units_%d" % self.num_layers],
                                      out_features))
        self.network = torch.nn.Sequential(*layers)

        return self

    def _add_layer(self, layers: List[torch.nn.Module], in_features: int, out_features: int,
                   layer_id: int) -> None:
        """
        Dynamically add a layer given the in->out specification

        Args:
            layers (List[nn.Module]): The list where all modules are added
            in_features (int): input dimensionality of the new layer
            out_features (int): output dimensionality of the new layer

        """
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(MLPNet.get_activations_dict()[self.activation]())
        if self.use_dropout:
            layers.append(torch.nn.Dropout(self.config["dropout_%d" % layer_id]))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'MLP',
            'name': 'Multi Layer Perceptron',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        min_mlp_layers: int = 1,
                                        max_mlp_layers: int = 15,
                                        dropout: bool = True,
                                        min_num_units: int = 10,
                                        max_num_units: int = 1024,
                                        ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The number of hidden layers the network will have.
        # Layer blocks are meant to have the same architecture, differing only
        # by the number of units
        num_layers = UniformIntegerHyperparameter(
            "num_layers", min_mlp_layers, max_mlp_layers, default_value=5)

        activation = CategoricalHyperparameter(
            "activation", choices=list(MLPNet.get_activations_dict().keys())
        )
        cs.add_hyperparameters([num_layers, activation])

        # We can have dropout in the network for
        # better generalization
        if dropout:
            use_dropout = CategoricalHyperparameter(
                "use_dropout", choices=[True, False])
            cs.add_hyperparameters([use_dropout])

        for i in range(1, max_mlp_layers + 1):
            n_units_hp = UniformIntegerHyperparameter("num_units_%d" % i,
                                                      lower=min_num_units,
                                                      upper=max_num_units,
                                                      default_value=20)
            cs.add_hyperparameter(n_units_hp)

            if i > min_mlp_layers:
                # The units of layer i should only exist
                # if there are at least i layers
                cs.add_condition(
                    CS.GreaterThanCondition(
                        n_units_hp, num_layers, i - 1
                    )
                )

            if dropout:
                dropout_hp = UniformFloatHyperparameter(
                    "dropout_%d" % i,
                    lower=0.0,
                    upper=0.8,
                    default_value=0.5
                )
                cs.add_hyperparameter(dropout_hp)
                dropout_condition_1 = CS.EqualsCondition(dropout_hp, use_dropout, True)

                if i > min_mlp_layers:
                    dropout_condition_2 = CS.GreaterThanCondition(dropout_hp, num_layers, i - 1)
                    cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
                else:
                    cs.add_condition(dropout_condition_1)

        return cs
