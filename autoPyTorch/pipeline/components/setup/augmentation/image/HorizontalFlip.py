from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class HorizontalFlip(BaseImageAugmenter):
    def __init__(self, p: float, random_state: Optional[int, np.random.RandomState] = None):
        super().__init__()
        self.random_state = random_state
        self.p = p

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter: Augmenter = iaa.Fliplr(p=self.p)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        X.update({'horizontal_flip': self.augmenter})
        return X

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        p = UniformFloatHyperparameter('p', lower=0, upper=1, default_value=0.5)
        cs.add_hyperparameter(p)
        return cs