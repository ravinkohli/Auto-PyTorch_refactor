from typing import Tuple, Optional, Dict, Any
from torch import nn
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.pipeline.components.setup.network.head.base_head import BaseHead


class FullyConvolutionalHead(BaseHead):
    supported_tasks = {"image_classification"}

    def build_head(self, input_shape: Tuple[int, int, int], output_shape: Tuple[int]) -> nn.Module:
        # TODO: improve this
        layers = []
        layers.append(nn.Conv2d(in_channels=input_shape[0],
                                out_channels=output_shape[0],
                                kernel_size=1))
        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        return nn.Sequential(*layers)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'FullyConvolutionalHead',
            'name': 'FullyConvolutionalHead',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None) -> ConfigurationSpace:
        if dataset_properties["task_type"] not in FullyConvolutionalHead.supported_tasks:
            raise ValueError(f"Unsupported task type {dataset_properties['task_type']}")

        cs = ConfigurationSpace()

        return cs
