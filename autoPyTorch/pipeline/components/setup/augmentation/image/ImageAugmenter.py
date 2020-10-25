import os
from collections import OrderedDict as OD
from typing import Any, Dict, Optional, OrderedDict, Union

from ConfigSpace.configuration_space import (
    ConfigurationSpace,
    Configuration
)

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

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


def get_components() -> OrderedDict[str, BaseImageAugmenter]:
    """Returns the available augmenter components

    Args:
        None

    Returns:
        Dict[str, BaseImageAugmenter]: all BaseImageAugmenter components available
            as choices
    """
    components = OD()
    components.update(_augmenters)
    components.update(_addons.components)
    return components


class ImageAugmenter(BaseImageAugmenter):

    def __init__(self, random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__()
        self.available_augmenters = get_components()  # type: OrderedDict[str, BaseImageAugmenter]
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter = iaa.Sequential([augmenter.fit(X).get_image_augmenter() for _, augmenter in self.available_augmenters.items()])
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'image_augmenter': self})
        return X

    def set_hyperparameters(self,
                            configuration: Configuration,
                            init_params: Optional[Dict[str, Any]] = None
                            ) -> 'ImageAugmenter':
        """
        Applies a configuration to the given component.
        This method translate a hierarchical configuration key,
        to an actual parameter of the autoPyTorch component.

        Args:
            configuration (Configuration): which configuration to apply to
                the chosen component
            init_params (Optional[Dict[str, any]]): Optional arguments to
                initialize the chosen component

        Returns:
            self: returns an instance of self
        """
        for name, augmenter in self.available_augmenters.items():
            new_params = {}

            params = configuration.get_dictionary()

            for param, value in params.items():
                if name in param:
                    param = param.replace(name, '').replace(':', '')
                    new_params[param] = value

            if init_params is not None:
                for param, value in init_params.items():
                    if name in param:
                        param = param.replace(name, '').replace(':', '')
                        new_params[param] = value

            new_params['random_state'] = self.random_state

            self.available_augmenters[name] = augmenter(**new_params)

        return self

    def get_hyperparameter_search_space(self,
                                        dataset_properties: Optional[Dict[str, str]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = dict()

        # add child hyperparameters
        for name in self.available_augmenters.keys():
            preprocessor_configuration_space = self.available_augmenters[name].\
                get_hyperparameter_search_space(dataset_properties)
            cs.add_configuration_space(name, preprocessor_configuration_space)

        return cs
