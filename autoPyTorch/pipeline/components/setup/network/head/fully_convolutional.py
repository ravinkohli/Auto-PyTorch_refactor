from typing import Tuple, Optional, Dict, Any
import torch
from torch import nn
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.pipeline.components.setup.network.head.base_head import BaseHead


class _FullyConvolutionalHead(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 output_shape: Tuple[int, ...]):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=input_shape[0],
                                out_channels=output_shape[0],
                                kernel_size=1))
        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        return self.head(x).view(B, -1)


class FullyConvolutionalHead(BaseHead):
    supported_tasks = {"image_classification", "image_regression"}

    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        return _FullyConvolutionalHead(input_shape=input_shape,
                                       output_shape=output_shape)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'FullyConvolutionalHead',
            'name': 'FullyConvolutionalHead',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        return cs
