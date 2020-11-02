from collections import OrderedDict
from typing import Type

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
)
from autoPyTorch.pipeline.components.setup.network.backbone.base_backbone import BaseBackbone
from autoPyTorch.pipeline.components.setup.network.backbone.image import ConvNetBackbone, DenseNetBackbone
from autoPyTorch.pipeline.components.setup.network.backbone.tabular import ResNetBackbone, ShapedMLPBackbone, \
    MLPBackbone
from autoPyTorch.pipeline.components.setup.network.backbone.time_series import TCNBackbone, InceptionTimeBackbone

_backbones = {
    ConvNetBackbone.get_name(): ConvNetBackbone,
    DenseNetBackbone.get_name(): DenseNetBackbone,
    ResNetBackbone.get_name(): ResNetBackbone,
    ShapedMLPBackbone.get_name(): ShapedMLPBackbone,
    MLPBackbone.get_name(): MLPBackbone,
    TCNBackbone.get_name(): TCNBackbone,
    InceptionTimeBackbone.get_name(): InceptionTimeBackbone
}
_addons = ThirdPartyComponents(BaseBackbone)


def add_backbone(backbone: BaseBackbone) -> None:
    _addons.add_component(backbone)


def get_available_backbones() -> OrderedDict[str, Type[BaseBackbone]]:
    backbones = OrderedDict()
    backbones.update(_backbones)
    backbones.update(_addons.components)
    return backbones
