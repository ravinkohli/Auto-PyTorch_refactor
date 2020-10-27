from typing import Any, Dict, Optional, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class Resize(BaseImageAugmenter):
    def __init__(self, interpolation: str = 'linear', use_augmenter: bool = True,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__(use_augmenter=use_augmenter)
        self.random_state = random_state
        self.interpolation = interpolation

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.check_requirements(X, y)
        if self.use_augmenter:
            self.augmenter: Augmenter = iaa.Resize(size=(X['image_height'], X['image_width']),
                                                   interpolation=self.interpolation, name=self.get_properties()['name'])

        return self

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        super().check_requirements(X, y)
        if 'image_height' not in X.keys():
            raise ValueError("Image height (image_height) not found in fit dictionary ")

        if 'image_width' not in X.keys():
            raise ValueError("Image width (image_width) not found in fit dictionary ")

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        interpolation = CategoricalHyperparameter('interpolation', choices=['nearest', 'linear', 'area', 'cubic'],
                                                  default_value='linear')
        use_augmenter = CategoricalHyperparameter('use_augmenter', choices=[True, False])
        cs.add_hyperparameters([interpolation, use_augmenter])

        # only add hyperparameters to configuration space if we are using the augmenter
        cs.add_condition(CS.EqualsCondition(interpolation, use_augmenter, True))

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'RandomAffine'}
