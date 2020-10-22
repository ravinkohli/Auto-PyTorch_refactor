from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter
)

import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class RandomAffine(BaseImageAugmenter):
    def __init__(self, scale_min: float, scale_offset: float,
                 translate_percent_min: float, translate_percent_offset: float,
                 shear: int, rotate: int, cval: int, mode: str,
                 random_state: Optional[int, np.random.RandomState] = None):
        super().__init__()
        self.random_state = random_state
        self.scale = (scale_min, scale_min + scale_offset)
        self.translate_percent = (translate_percent_min, translate_percent_min + translate_percent_offset)
        self.shear = (-shear, shear)
        self.rotate = (-rotate, rotate)
        self.cval = cval  # TODO maybe a range that cval can sample per image
        self.mode = mode

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter: Augmenter = iaa.Affine(scale=self.scale, translate_percent=self.translate_percent, rotate=self.rotate,
                                    shear=self.shear, cval=self.cval)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        X.update({'affine': self.augmenter})
        return X

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        scale_min = UniformFloatHyperparameter('scale_x_min', lower=0, upper=0.99, default_value=0.5)
        scale_offset = UniformFloatHyperparameter('scale_x_min', lower=0, upper=0.99, default_value=0.5)
        translate_percent_min = UniformFloatHyperparameter('scale_x_min', lower=0, upper=0.99, default_value=0.5)
        translate_percent_offset = UniformFloatHyperparameter('scale_x_min', lower=0, upper=0.99, default_value=0.5)
        shear = UniformIntegerHyperparameter('shear_x_min', lower=0, upper=45, default_value=45)
        rotate = UniformIntegerHyperparameter('shear_x_min', lower=0, upper=360, default_value=45)
        cval = UniformIntegerHyperparameter('cval', lower=0, upper=255, default_value=0)
        mode = CategoricalHyperparameter('mode', choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'])

        cs.add_hyperparameters([scale_min, scale_offset, translate_percent_min, translate_percent_offset])
        cs.add_hyperparameters([shear, rotate, cval, mode])

        return cs