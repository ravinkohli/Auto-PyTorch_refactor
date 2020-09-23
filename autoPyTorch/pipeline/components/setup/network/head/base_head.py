from typing import Set, Any, Dict, Tuple
from torch import nn

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent, BaseEstimator
from autoPyTorch.pipeline.components.setup.network.head import fully_convolutional


class BaseHead(autoPyTorchComponent):
    supported_tasks: Set = set()

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.head = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        input_shape = X["backbone_output_shape"]
        output_shape = X["head_output_shape"]
        self.head = self.build_head(input_shape=input_shape, output_shape=output_shape)
        return self

    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        raise NotImplementedError


def get_available_heads() -> Set[BaseHead]:
    return {fully_convolutional.ImageHead}
