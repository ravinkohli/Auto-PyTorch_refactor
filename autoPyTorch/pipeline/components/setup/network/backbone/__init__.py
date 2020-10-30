import os
from typing import Dict
from collections import OrderedDict

from autoPyTorch.pipeline.components.setup.network.backbone.base_backbone import BaseBackbone
from autoPyTorch.pipeline.components.setup.network.backbone.tabular import MLPBackbone, ShapedMLPBackbone
from autoPyTorch.pipeline.components.setup.network.backbone.time_series import InceptionTimeBackbone, TCNBackbone

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    find_components,
)

_directory = os.path.split(__file__)[0]
_backbones = find_components(__package__,
                             _directory,
                             BaseBackbone)
_addons = ThirdPartyComponents(BaseBackbone)


def add_backbone(backbone: BaseBackbone):
    _addons.add_component(backbone)


def get_available_backbones() -> Dict[str, BaseBackbone]:
    backbones = OrderedDict()
    backbones.update(_backbones)
    backbones.update(_addons.components)
    return backbones
