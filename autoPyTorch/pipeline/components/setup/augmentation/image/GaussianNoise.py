from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class GaussianNoise(BaseImageAugmenter):
    def __init__(self, mean_min: float, sigma_min: float,
                 mean_offset: float, sigma_offset: float,
                 random_state: Optional[int, np.random.RandomState] = None):
        super().__init__()
        self.random_state = random_state
        self.mean = (mean_min, mean_min + mean_offset)
        self.sigma = (sigma_min, sigma_min + sigma_offset)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter: Augmenter = iaa.AdditiveGaussianNoise()

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        X.update({'gaussian_noise': self.augmenter})
        return X

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        mean_min = UniformFloatHyperparameter('mean_min', lower=0, upper=1, default_value=0)
        mean_offset = UniformFloatHyperparameter('mean_offset', lower=0, upper=1, default_value=0)
        sigma_min = UniformFloatHyperparameter('sigma_min', lower=0, upper=3, default_value=0)
        sigma_offset = UniformFloatHyperparameter('sigma_offset', lower=0, upper=3, default_value=0.3)
        cs.add_hyperparameters([mean_min, mean_offset, sigma_min, sigma_offset])
        return cs