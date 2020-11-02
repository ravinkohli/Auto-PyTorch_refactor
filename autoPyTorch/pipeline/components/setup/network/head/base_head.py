from abc import abstractmethod
from typing import Set, Any, Dict, Tuple

import torch.nn as nn

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent, BaseEstimator


class BaseHead(autoPyTorchComponent):
    supported_tasks: Set = set()

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.head: nn.Module = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Not used. Just for API compatibility.
        """
        return self

    @abstractmethod
    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        """

        Builds the head module

        :param input_shape: shape of the input
        :param output_shape: shape of the output
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        return cls.get_properties()["shortname"]
