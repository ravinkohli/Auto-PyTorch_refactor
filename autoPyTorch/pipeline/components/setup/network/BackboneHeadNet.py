from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)
from torch import nn

from autoPyTorch.pipeline.components.setup.network.backbone import get_available_backbones, BaseBackbone
from autoPyTorch.pipeline.components.setup.network.base_network import BaseNetworkComponent
from autoPyTorch.pipeline.components.setup.network.head import get_available_heads, BaseHead


class BackboneHeadNet(BaseNetworkComponent):
    """
    Implementation of a dynamic network, that consists of a backbone and a head
    """

    def __init__(
            self,
            random_state: Optional[np.random.RandomState] = None,
            **kwargs: Any
    ):
        super().__init__(
            random_state=random_state,
        )
        self.config = kwargs
        self._backbones = get_available_backbones()
        self._heads = get_available_heads()
        self._backbones = get_available_backbones()
        self._heads = get_available_heads()

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {
            "shortname": "BackboneHeadNet",
            "name": "BackboneHeadNet",
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict[str, str]] = None,
                                        **kwargs: Any) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        backbones: Dict[str, Type[BaseBackbone]] = get_available_backbones()
        heads: Dict[str, Type[BaseHead]] = get_available_heads()

        if dataset_properties is not None and "task_type" in dataset_properties:
            # filter backbones and heads for those who support the current task type
            task = dataset_properties["task_type"]
            backbones = {name: backbone for name, backbone in backbones.items() if task in backbone.supported_tasks}
            heads = {name: head for name, head in heads.items() if task in head.supported_tasks}

        backbone_hp = CategoricalHyperparameter("backbone", choices=backbones.keys())
        head_hp = CategoricalHyperparameter("head", choices=heads.keys())
        cs.add_hyperparameters([backbone_hp, head_hp])

        # for each backbone and head, add a conditional search space if this backbone or head is chosen
        for backbone_name in backbones.keys():
            backbone_cs = backbones[backbone_name].get_hyperparameter_search_space(dataset_properties)
            cs.add_configuration_space(backbone_name,
                                       backbone_cs,
                                       parent_hyperparameter={"parent": backbone_hp, "value": backbone_name})

        for head_name in heads.keys():
            head_cs: ConfigurationSpace = heads[head_name].get_hyperparameter_search_space(dataset_properties)
            cs.add_configuration_space(head_name,
                                       head_cs,
                                       parent_hyperparameter={"parent": head_hp, "value": head_name})
        return cs

    def build_network(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        """This method returns a pytorch network, that is dynamically built

            a self.config that is network specific, and contains the additional
            configuration hyperparameters to build a domain specific network
        """
        backbone_name = self.config["backbone"]
        head_name = self.config["head"]
        Backbone = self._backbones[backbone_name]
        Head = self._heads[head_name]

        backbone = Backbone(**{k.replace(backbone_name, "").replace(":", ""): v
                               for k, v in self.config.items() if
                               k.startswith(backbone_name)})
        backbone_module = backbone.build_backbone(input_shape=input_shape)
        backbone_output_shape = backbone.get_output_shape(input_shape=input_shape)

        head = Head(**{k.replace(head_name, "").replace(":", ""): v
                       for k, v in self.config.items() if
                       k.startswith(head_name)})
        head_module = head.build_head(input_shape=backbone_output_shape, output_shape=output_shape)

        return nn.Sequential(backbone_module, head_module)

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        info = vars(self)
        # Remove unwanted info
        info.pop('network', None)
        info.pop('random_state', None)
        return f"BackboneHeadNet: {self.config['backbone']} -> {self.config['head']} ({str(info)})"
