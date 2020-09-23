from typing import Tuple, Optional, Dict, Any
from torch import nn
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from autoPyTorch.pipeline.components.setup.network.head.base_head import BaseHead


class FullyConnectedHead(BaseHead):
    supported_tasks = {"tabular_classification", "tabular_regression",
                       "image_classification", "image_regression",
                       "time_series_classification", "time_series_regression"}

    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int]) -> nn.Module:
        # TODO: improve this
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.Linear(np.prod(input_shape).item(), np.prod(output_shape).item()))
        return nn.Sequential(*layers)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'FullyConnectedHead',
            'name': 'FullyConnectedHead',
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None) -> ConfigurationSpace:
        if dataset_properties["task_type"] not in FullyConnectedHead.supported_tasks:
            raise ValueError(f"Unsupported task type {dataset_properties['task_type']}")

        cs = ConfigurationSpace()

        return cs
