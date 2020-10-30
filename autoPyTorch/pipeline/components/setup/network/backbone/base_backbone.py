from abc import abstractmethod
from typing import Set, Any, Dict, Tuple

import torch
from torch import nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.base_component import (
    autoPyTorchComponent,
)


class BaseBackbone(autoPyTorchComponent):
    supported_tasks: Set = set()

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.backbone = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Not used. Just for API compatibility.
        """
        return self

    @abstractmethod
    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        raise NotImplementedError()

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Run a dummy forward pass to get the output shape of the backbone

        :param input_shape: shape of the input
        :return: output_shape
        """
        placeholder = torch.randn((1, *input_shape), dtype=torch.float)
        with torch.no_grad():
            output = self.backbone(placeholder)
        return tuple(output.shape[1:])
