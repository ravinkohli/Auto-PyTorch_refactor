import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import (
    ConfigurationSpace,
    Configuration
)

import imgaug.augmenters as iaa

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
    def __init__(self, random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__()
        self.available_augmenters = Optional[Dict[str, BaseImageAugmenter]] = None
        self.random_state = random_state

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

    def get_available_components(
        self,
        dataset_properties: Optional[Dict[str, str]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, BaseImageAugmenter]:
        """
        Wrapper over get components to incorporate include/exclude
        user specification

        Args:
            dataset_properties (Optional[Dict[str, str]]): Describes the dataset to work on
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip

        Results:
            Dict[str, autoPyTorchComponent]: A dictionary with valid components for this
                choice object

        """
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            components_dict[name] = available_comp[name]

        return components_dict

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter = iaa.Sequential([augmenter for _, augmenter in self.available_augmenters.items()])
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'image_augmenter': self.augmenter})
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

            self.available_augmenters.update(name, augmenter(**new_params))

        return self

    def get_hyperparameter_search_space(self,
        dataset_properties: Optional[Dict[str, str]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = dict()
        self.available_augmenters = self.get_available_components(dataset_properties=dataset_properties,
                                                                  include=include,
                                                                  exclude=exclude)

        # add child hyperparameters
        for name in self.available_augmenters.keys():
            preprocessor_configuration_space = self.available_augmenters[name].\
                get_hyperparameter_search_space(dataset_properties)
            cs.add_configuration_space(name, preprocessor_configuration_space)

        return ConfigurationSpace()
