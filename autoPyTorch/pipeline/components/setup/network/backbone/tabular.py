from typing import Any, Dict, List, Optional, Tuple

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import torch
from torch import nn

from autoPyTorch.pipeline.components.setup.network.backbone.base_backbone import BaseBackbone
from autoPyTorch.pipeline.components.setup.network.utils import get_shaped_neuron_counts

_activations = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}


class MLP(BaseBackbone):
    """
    This component automatically creates a Multi Layer Perceptron based on a given config.

    This MLP allows for:
        - Different number of layers
        - Specifying the activation. But this activation is shared among layers
        - Using or not dropout
        - Specifying the number of units per layers
    """
    supported_tasks = {"tabular_classification", "tabular_regression"}

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """Returns the actual PyTorch model, that is dynamically created
                from a self.config object.

                self.config is a dictionary created form a given config in the config space.
                It contains the necessary information to build a network.
                """
        layers = list()  # type: List[torch.nn.Module]
        in_features = input_shape[0]

        self._add_layer(layers, in_features, self.config['num_units_1'], 1)

        for i in range(2, self.config['num_groups'] + 1):
            self._add_layer(layers, self.config["num_units_%d" % (i - 1)],
                            self.config["num_units_%d" % i], i)
        network = torch.nn.Sequential(*layers)
        return network

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return (self.config["num_units_%d" % self.config["num_groups"]],)

    def _add_layer(self, layers: List[torch.nn.Module], in_features: int, out_features: int,
                   layer_id: int) -> None:
        """
        Dynamically add a layer given the in->out specification

        Args:
            layers (List[nn.Module]): The list where all modules are added
            in_features (int): input dimensionality of the new layer
            out_features (int): output dimensionality of the new layer

        """
        layers.append(nn.Linear(in_features, out_features))
        layers.append(_activations[self.config["activation"]]())
        if self.config['use_dropout']:
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
        num_groups = UniformIntegerHyperparameter(
            "num_groups", min_mlp_layers, max_mlp_layers, default_value=5)

        activation = CategoricalHyperparameter(
            "activation", choices=list(_activations.keys())
        )
        cs.add_hyperparameters([num_groups, activation])

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
                        n_units_hp, num_groups, i - 1
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
                    dropout_condition_2 = CS.GreaterThanCondition(dropout_hp, num_groups, i - 1)
                    cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
                else:
                    cs.add_condition(dropout_condition_1)

        return cs


class ShapedMLP(BaseBackbone):
    """
        Implementation of a Shaped MLP -- an MLP with the number of units
        arranged so that a given shape is honored
    """
    supported_tasks = {"tabular_classification", "tabular_regression"}

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        layers = list()  # type: List[torch.nn.Module]
        in_features = input_shape[0]
        out_features = self.config["output_dim"]
        neuron_counts = get_shaped_neuron_counts(self.config['mlp_shape'],
                                                 in_features,
                                                 out_features,
                                                 self.config['max_units'],
                                                 self.config['num_groups'])
        if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
            dropout_shape = get_shaped_neuron_counts(
                self.config['mlp_shape'], 0, 0, 1000, self.config['num_groups']
            )

        previous = in_features
        for i in range(self.config['num_groups'] - 1):
            if i >= len(neuron_counts):
                break
            if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
                dropout = dropout_shape[i] / 1000 * self.config["max_dropout"]
            else:
                dropout = 0.0
            self._add_layer(layers, previous, neuron_counts[i], dropout)
            previous = neuron_counts[i]
        return torch.nn.Sequential(*layers)

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return (self.config["output_dim"],)

    def _add_layer(self, layers: List[torch.nn.Module],
                   in_features: int, out_features: int, dropout: float
                   ) -> None:
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(_activations["activation"]())
        if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
            layers.append(torch.nn.Dropout(dropout))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'ShapedMLP',
            'name': 'Shaped Multi Layer Perceptron',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        min_num_gropus: int = 1,
                                        max_num_groups: int = 15,
                                        min_num_units: int = 10,
                                        max_num_units: int = 1024,
                                        ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The number of groups that will compose the resnet. That is,
        # a group can have N Resblock. The M number of this N resblock
        # repetitions is num_groups
        num_groups = UniformIntegerHyperparameter(
            "num_groups", lower=min_num_gropus, upper=max_num_groups, default_value=5)

        mlp_shape = CategoricalHyperparameter('mlp_shape', choices=[
            'funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'
        ])

        activation = CategoricalHyperparameter(
            "activation", choices=list(_activations.keys())
        )

        max_units = UniformIntegerHyperparameter(
            "max_units",
            lower=min_num_units,
            upper=max_num_units,
        )

        output_dim = UniformIntegerHyperparameter(
            "output_dim",
            lower=min_num_units,
            upper=max_num_units
        )

        cs.add_hyperparameters([num_groups, activation, mlp_shape, max_units, output_dim])

        # We can have dropout in the network for
        # better generalization
        use_dropout = CategoricalHyperparameter(
            "use_dropout", choices=[True, False])
        max_dropout = UniformFloatHyperparameter("max_dropout", lower=0.0, upper=1.0)
        cs.add_hyperparameters([use_dropout, max_dropout])
        cs.add_condition(CS.EqualsCondition(max_dropout, use_dropout, True))

        return cs
