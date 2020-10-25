from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class RandomAffine(BaseImageAugmenter):
    def __init__(self, scale_min: float = 0, scale_offset: float = 0.2,
                 translate_percent_min: float = 0, translate_percent_offset: float = 0.3,
                 shear: int = 30, rotate: int = 45, random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__()
        self.random_state = random_state
        self.scale = (scale_min, scale_min + scale_offset)
        self.translate_percent = (translate_percent_min, translate_percent_min + translate_percent_offset)
        self.shear = (-shear, shear)
        self.rotate = (-rotate, rotate)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter: Augmenter = iaa.Affine(scale=self.scale, translate_percent=self.translate_percent,
                                               rotate=self.rotate, shear=self.shear, mode='symmetric')

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        scale_min = UniformFloatHyperparameter('scale_min', lower=0, upper=0.99, default_value=0)
        scale_offset = UniformFloatHyperparameter('scale_offset', lower=0, upper=0.99, default_value=0.2)
        translate_percent_min = UniformFloatHyperparameter('translate_percent_min', lower=0, upper=0.99, default_value=0)
        translate_percent_offset = UniformFloatHyperparameter('translate_percent_offset', lower=0, upper=0.99,
                                                              default_value=0.3)
        shear = UniformIntegerHyperparameter('shear', lower=0, upper=45, default_value=30)
        rotate = UniformIntegerHyperparameter('rotate', lower=0, upper=360, default_value=45)

        cs.add_hyperparameters([scale_min, scale_offset, translate_percent_min, translate_percent_offset])
        cs.add_hyperparameters([shear, rotate])

        return cs