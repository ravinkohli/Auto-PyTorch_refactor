from abc import abstractmethod
from typing import Set, Any, Dict, Tuple

from torch import nn

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent, BaseEstimator


class BaseHead(autoPyTorchComponent):
    supported_tasks: Set = set()

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.head = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Not used. Just for API compatibility.
        """
        return self

    @abstractmethod
    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        raise NotImplementedError()
