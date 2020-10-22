from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class RandomCutout(BaseImageAugmenter):
    def __init__(self, nb_iterartions_min: int, nb_iterartions_offset: int,
                 position: str, size_min: float, size_offset: float, squared: bool, fill_mode: str, cval: int = 0,
                 fill_per_channel: bool = True, random_state: Optional[int, np.random.RandomState] = None):
        super().__init__()
        self.nb_iterartions = (nb_iterartions_min, nb_iterartions_min + nb_iterartions_offset)
        self.position = position
        self.size = (size_min, size_min + size_offset)
        self.squared = squared
        self.fill_mode = fill_mode
        self.cval = cval
        self.fill_per_channel = fill_per_channel
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter: Augmenter = iaa.Cutout(nb_iterations=self.nb_iterartions, position=self.position,
                                               size=self.size, squared=self.squared, fill_mode=self.fill_mode,
                                               cval=self.cval, fill_per_channel=self.fill_per_channel,
                                               random_state=self.random_state)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'cutout': self.augmenter})
        return X

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        nb_iterartions_min = UniformIntegerHyperparameter('nb_iterartions_min', lower=0, upper=4, default=2)
        nb_iterartions_offset = UniformIntegerHyperparameter('nb_iterartions_offset', lower=0, upper=4, default=2)
        position = CategoricalHyperparameter('position', choices=['uniform', 'normal', 'center', 'left-top',
                                                                  'left-center', 'left-bottom', 'center-top',
                                                                  'center-center', 'center-bottom', 'right-top',
                                                                  'right-center', 'right-bottom'],
                                             default_value='normal')
        cs.add_hyperparameters([nb_iterartions_min, nb_iterartions_offset, position])
        size_min = UniformFloatHyperparameter('size_min', lower=0, upper=0.5, default_value=0.1, log=True)
        size_offset = UniformFloatHyperparameter('size_offset', lower=0, upper=0.5, default_value=0.5)
        squared = CategoricalHyperparameter('squared', choices=[True, False], default_value=False)
        fill_mode = CategoricalHyperparameter('fill_mode', choices=['constant', 'gaussian'])
        cval = UniformIntegerHyperparameter('cval', lower=0, upper=255, default=0)
        fill_per_channel = CategoricalHyperparameter('fill_per_channel', choices=[True, False], default_value=False)
        cs.add_hyperparameters([size_min, size_offset, squared, fill_mode, cval, fill_per_channel])
        return cs