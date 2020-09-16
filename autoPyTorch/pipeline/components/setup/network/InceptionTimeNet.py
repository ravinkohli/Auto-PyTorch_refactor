# Code inspired by https://github.com/hfawaz/InceptionTime
# Paper: https://arxiv.org/pdf/1909.04939.pdf
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch import nn

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from autoPyTorch.pipeline.components.setup.network.base_network import BaseNetworkComponent


class _InceptionBlock(nn.Module):
    def __init__(self, n_inputs, n_filters, kernel_size, bottleneck=None):
        super(_InceptionBlock, self).__init__()
        self.n_filters = n_filters
        self.bottleneck = None \
            if bottleneck is None \
            else nn.Conv1d(n_inputs, bottleneck, kernel_size=1)
        kernel_sizes = [kernel_size // (2 ** i) for i in range(3)]
        n_inputs = n_inputs if bottleneck is None else bottleneck
        # create 3 conv layers with different kernel sizes which are applied in parallel
        self.pad1 = nn.ConstantPad1d(
            padding=self.padding(kernel_sizes[0]), value=0)
        self.conv1 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[0])
        self.pad2 = nn.ConstantPad1d(
            padding=self.padding(kernel_sizes[1]), value=0)
        self.conv2 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[1])
        self.pad3 = nn.ConstantPad1d(
            padding=self.padding(kernel_sizes[2]), value=0)
        self.conv3 = nn.Conv1d(n_inputs, n_filters, kernel_sizes[2])
        # create 1 maxpool and conv layer which are also applied in parallel
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.convpool = nn.Conv1d(n_inputs, n_filters, 1)

        self.bn = nn.BatchNorm1d(4 * n_filters)

    def padding(self, kernel_size):
        if kernel_size % 2 == 0:
            return kernel_size // 2, kernel_size // 2 - 1
        else:
            return kernel_size // 2, kernel_size // 2

    def get_n_outputs(self):
        return 4 * self.n_filters

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        x1 = self.conv1(self.pad1(x))
        x2 = self.conv2(self.pad2(x))
        x3 = self.conv3(self.pad3(x))
        x4 = self.convpool(self.maxpool(x))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bn(x)
        return torch.relu(x)


class _ResidualBlock(nn.Module):
    def __init__(self, n_res_inputs, n_outputs):
        super(_ResidualBlock, self).__init__()
        self.shortcut = nn.Conv1d(n_res_inputs, n_outputs, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_outputs)

    def forward(self, x, res):
        shortcut = self.shortcut(res)
        shortcut = self.bn(shortcut)
        x += shortcut
        return torch.relu(x)


class _InceptionTime(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        n_inputs = in_features
        n_filters = self.config["num_filters"]
        bottleneck_size = self.config["bottleneck_size"]
        kernel_size = self.config["kernel_size"]
        n_res_inputs = in_features
        for i in range(self.config["num_blocks"]):
            block = _InceptionBlock(n_inputs=n_inputs,
                                    n_filters=n_filters,
                                    bottleneck=bottleneck_size,
                                    kernel_size=kernel_size)
            self.__setattr__(f"inception_block_{i}", block)

            # add a residual block after every 3 inception blocks
            if i % 3 == 2:
                n_res_outputs = block.get_n_outputs()
                self.__setattr__(f"residual_block_{i}", _ResidualBlock(n_res_inputs=n_res_inputs,
                                                                       n_outputs=n_res_outputs))
                n_res_inputs = n_res_outputs
            n_inputs = block.get_n_outputs()

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        fc_layers = [
            nn.Linear(in_features=block.get_n_outputs(), out_features=out_features)]
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        # swap sequence and feature dimensions for use with convolutional nets
        x = x.transpose(1, 2).contiguous()
        res = x
        for i in range(self.config["num_blocks"]):
            x = self.__getattr__(f"inception_block_{i}")(x)
            if i % 3 == 2:
                x = self.__getattr__(f"residual_block_{i}")(x, res)
                res = x
        x = self.global_avg_pool(x)
        x = x.permute(0, 2, 1)
        x = self.fc_layers(x).squeeze(dim=1)
        return x


class InceptionTime(BaseNetworkComponent):
    def __init__(self,
                 intermediate_activation: str,
                 final_activation: Optional[str],
                 random_state: Optional[np.random.RandomState] = None,
                 **kwargs) -> None:
        super().__init__(intermediate_activation, final_activation, random_state)
        self.config = kwargs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            "shortname": "InceptionTime",
            "name": "InceptionTime",
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        min_num_blocks: int = 1,
                                        max_num_blocks: int = 10,
                                        min_num_filters: int = 16,
                                        max_num_filters: int = 64,
                                        min_kernel_size: int = 32,
                                        max_kernel_size: int = 64,
                                        min_bottleneck_size: int = 16,
                                        max_bottleneck_size: int = 64,
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        num_blocks_hp = UniformIntegerHyperparameter("num_blocks",
                                                     lower=min_num_blocks,
                                                     upper=max_num_blocks)
        cs.add_hyperparameter(num_blocks_hp)

        num_filters_hp = UniformIntegerHyperparameter("num_filters",
                                                      lower=min_num_filters,
                                                      upper=max_num_filters)
        cs.add_hyperparameter(num_filters_hp)

        bottleneck_size_hp = UniformIntegerHyperparameter("bottleneck_size",
                                                          lower=min_bottleneck_size,
                                                          upper=max_bottleneck_size)
        cs.add_hyperparameter(bottleneck_size_hp)

        kernel_size_hp = UniformIntegerHyperparameter("kernel_size",
                                                      lower=min_kernel_size,
                                                      upper=max_kernel_size)
        cs.add_hyperparameter(kernel_size_hp)
        return cs

    def build_network(self, in_feature: int, out_features: int) -> torch.nn.Module:
        network = _InceptionTime(in_features=in_feature,
                                 out_features=out_features,
                                 config=self.config)
        return network
