import os
from typing import Dict
from collections import OrderedDict

from autoPyTorch.pipeline.components.setup.network.head.base_head import BaseHead
from autoPyTorch.pipeline.components.setup.network.head.fully_connected import FullyConnectedHead
from autoPyTorch.pipeline.components.setup.network.head.fully_convolutional import FullyConvolutionalHead

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    find_components,
)

_directory = os.path.split(__file__)[0]
_heads = find_components(__package__,
                         _directory,
                         BaseHead)
_addons = ThirdPartyComponents(BaseHead)


def add_head(head: BaseHead):
    _addons.add_component(head)


def get_available_heads() -> Dict[str, BaseHead]:
    heads = OrderedDict()
    heads.update(_heads)
    heads.update(_addons.components)
    return heads
