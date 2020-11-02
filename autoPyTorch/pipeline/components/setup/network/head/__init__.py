from collections import OrderedDict
from typing import Type

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents
)
from autoPyTorch.pipeline.components.setup.network.head.base_head import BaseHead
from autoPyTorch.pipeline.components.setup.network.head.fully_connected import FullyConnectedHead
from autoPyTorch.pipeline.components.setup.network.head.fully_convolutional import FullyConvolutionalHead

_heads = {
    FullyConnectedHead.get_name(): FullyConnectedHead,
    FullyConvolutionalHead.get_name(): FullyConvolutionalHead
}
_addons = ThirdPartyComponents(BaseHead)


def add_head(head: BaseHead) -> None:
    _addons.add_component(head)


def get_available_heads() -> OrderedDict[str, Type[BaseHead]]:
    heads = OrderedDict()
    heads.update(_heads)
    heads.update(_addons.components)
    return heads
