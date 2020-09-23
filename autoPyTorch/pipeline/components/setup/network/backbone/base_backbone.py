from typing import Set, Any, Dict, Tuple
import torch
from torch import nn

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent, BaseEstimator


class BaseBackbone(autoPyTorchComponent):
    supported_tasks: Set = set()

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.backbone = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        input_shape = X["input_shape"]
        self.backbone = self.build_backbone(input_shape=input_shape)
        return self

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        raise NotImplementedError

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Run a dummy forward pass to get the output shape of the backbone

        :param input_shape: shape of the input
        :return: output_shape
        """
        placeholder = torch.randn((1, *input_shape), dtype=torch.float)
        with torch.no_grad():
            output = self.backbone(placeholder)
        return output.shape
