from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class GaussianNoise(BaseImageAugmenter):
    def __init__(self, sigma_offset: float = 0.3,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__()
        self.random_state = random_state
        self.sigma = (0, sigma_offset)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter: Augmenter = iaa.AdditiveGaussianNoise(scale=self.sigma)
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        sigma_offset = UniformFloatHyperparameter('sigma_offset', lower=0, upper=3, default_value=0.3)
        cs.add_hyperparameter(sigma_offset)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'GaussianNoise'}
