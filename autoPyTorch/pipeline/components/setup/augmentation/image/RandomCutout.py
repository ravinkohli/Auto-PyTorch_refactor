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
    def __init__(self, position: str, squared: bool, fill_mode: str, cval: int = 0,
                 fill_per_channel: bool = True, p: float = 0.5,
                 random_state: Optional[int, np.random.RandomState] = None):
        super().__init__()
        self.position = position
        self.squared = squared
        self.fill_mode = fill_mode
        self.cval = cval
        self.fill_per_channel = fill_per_channel
        self.p = p
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter: Augmenter = iaa.Sometimes(self.p, iaa.Cutout(nb_iterations=(1, 10), position=self.position,
                                                  size=(0.1, 0.5), squared=self.squared, fill_mode=self.fill_mode,
                                                  cval=self.cval, fill_per_channel=self.fill_per_channel,
                                                  random_state=self.random_state))

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        position = CategoricalHyperparameter('position', choices=['uniform', 'normal', 'center', 'left-top',
                                                                  'left-center', 'left-bottom', 'center-top',
                                                                  'center-center', 'center-bottom', 'right-top',
                                                                  'right-center', 'right-bottom'],
                                             default_value='normal')
        squared = CategoricalHyperparameter('squared', choices=[True, False], default_value=False)
        fill_mode = CategoricalHyperparameter('fill_mode', choices=['constant', 'gaussian'])
        cval = UniformIntegerHyperparameter('cval', lower=0, upper=255, default=0)
        fill_per_channel = CategoricalHyperparameter('fill_per_channel', choices=[True, False], default_value=False)
        p = UniformFloatHyperparameter('p', lower=0.2, upper=1, default=0.5)
        cs.add_hyperparameters([position, squared, fill_mode, cval, fill_per_channel, p])
        return cs