import os
from collections import OrderedDict
from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
import numpy as np

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    find_components,
)
from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


augmenter_directory = os.path.split(__file__)[0]
_augmenters = find_components(__package__,
                            augmenter_directory,
                            BaseImageAugmenter)
_addons = ThirdPartyComponents(BaseImageAugmenter)


def add_augmenter(augmenter: BaseImageAugmenter) -> None:
    _addons.add_component(augmenter)


class ImageAugmenter(BaseImageAugmenter):
    def __init__(self, random_state: Optional[int, np.random.RandomState] = None, **kwargs):
        super().__init__()
        self.random_state = random_state
        self.probability_augmenters = list()
        for key, value in kwargs.items():
            self.probability_augmenters.append(value)

    def get_components(self) -> Dict[str, BaseImageAugmenter]:
        """Returns the available augmenter components

        Args:
            None

        Returns:
            Dict[str, BaseImageAugmenter]: all BaseImageAugmenter components available
                as choices
        """
        components = OrderedDict()
        components.update(_augmenters)
        components.update(_addons.components)
        return components

    # def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
    #     for i in range()
    #     return self
    #
    # @staticmethod
    # def get_hyperparameter_search_space(
    #     dataset_properties: Optional[Dict[str, str]] = None
    # ) -> ConfigurationSpace:
    #     return ConfigurationSpace()


