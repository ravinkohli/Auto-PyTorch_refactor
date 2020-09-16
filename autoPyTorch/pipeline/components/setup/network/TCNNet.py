# Chomp1d, TemporalBlock and TemporalConvNet copied from
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py, Carnegie Mellon University Locus Labs
# Paper: https://arxiv.org/pdf/1803.01271.pdf
from typing import Optional, Dict, Any

import numpy as np
from torch import nn
from torch.nn.utils import weight_norm

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

from autoPyTorch.pipeline.components.setup.network.base_network import BaseNetworkComponent


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(_Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(_TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        # self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(_TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [_TemporalBlock(in_channels,
                                      out_channels,
                                      kernel_size,
                                      stride=1,
                                      dilation=dilation_size,
                                      padding=(kernel_size - 1) * dilation_size,
                                      dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNNet(BaseNetworkComponent):
    def __init__(self,
                 intermediate_activation: str,
                 final_activation: Optional[str],
                 random_state: Optional[np.random.RandomState] = None,
                 **kwargs: Any) -> None:
        super().__init__(intermediate_activation, final_activation, random_state)
        self.config = kwargs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            "shortname": "TCN",
            "name": "Temporal Convolutional Network",
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_num_blocks: int = 1,
                                        max_num_blocks: int = 10,
                                        min_num_filters: int = 4,
                                        max_num_filters: int = 64,
                                        min_kernel_size: int = 4,
                                        max_kernel_size: int = 64,
                                        min_dropout: float = 0.0,
                                        max_dropout: float = 0.5
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        num_blocks_hp = UniformIntegerHyperparameter("num_blocks",
                                                     lower=min_num_blocks,
                                                     upper=max_num_blocks)
        cs.add_hyperparameter(num_blocks_hp)

        kernel_size_hp = UniformIntegerHyperparameter("kernel_size",
                                                      lower=min_kernel_size,
                                                      upper=max_kernel_size)
        cs.add_hyperparameter(kernel_size_hp)

        use_dropout_hp = CategoricalHyperparameter("use_dropout",
                                                   choices=[True, False])
        cs.add_hyperparameter(use_dropout_hp)

        dropout_hp = UniformFloatHyperparameter("dropout",
                                                lower=min_dropout,
                                                upper=max_dropout)
        cs.add_hyperparameter(dropout_hp)
        cs.add_condition(CS.EqualsCondition(dropout_hp, use_dropout_hp, True))

        for i in range(0, max_num_blocks):
            num_filters_hp = UniformIntegerHyperparameter(f"num_filters_{i}",
                                                          lower=min_num_filters,
                                                          upper=max_num_filters)
            cs.add_hyperparameter(num_filters_hp)
            if i >= min_num_blocks:
                cs.add_condition(CS.GreaterThanCondition(
                    num_filters_hp, num_blocks_hp, i))

        return cs

    def build_network(self, in_feature: int, out_features: int) -> nn.Module:
        num_channels = [self.config["num_filters_0"]]
        for i in range(1, self.config["num_blocks"]):
            num_channels.append(self.config[f"num_filters_{i}"])
        tcn = _TemporalConvNet(in_feature,
                               num_channels,
                               kernel_size=self.config["kernel_size"],
                               dropout=self.config["dropout"] if self.config["use_dropout"] else 0.0
                               )
        fc_layers = [nn.Linear(in_features=num_channels[-1],
                               out_features=out_features)]
        network = nn.Sequential(tcn, *fc_layers)
        return network
