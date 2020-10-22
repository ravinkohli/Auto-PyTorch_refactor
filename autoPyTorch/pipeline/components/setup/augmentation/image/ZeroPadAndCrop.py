from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter
)

import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class ZeroPadAndCrop(BaseImageAugmenter):
    def __init__(self, percent: float, random_state: Optional[int, np.random.RandomState] = None):
        super().__init__()
        self.random_state = random_state
        self.percent = percent
        self.pad_augmenter: Optional[Augmenter] = None
        self.crop_augmenter: Optional[Augmenter] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.pad_augmenter = iaa.Pad(percent=self.percent, keep_size=False)
        self.crop_augmenter = iaa.Crop(percent=self.percent, keep_size=False)
        self.augmenter: Augmenter = iaa.Sequential([
            self.pad_augmenter,
            self.crop_augmenter
        ])

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'zero_pad_crop': self.augmenter})
        return X

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        percent = UniformFloatHyperparameter('percent', lower=0, upper=0.5, default_value=0.1)
        cs.add_hyperparameter(percent)
        return cs