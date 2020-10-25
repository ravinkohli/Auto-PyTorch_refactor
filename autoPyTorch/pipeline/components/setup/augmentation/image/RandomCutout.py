from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class RandomCutout(BaseImageAugmenter):
    def __init__(self, p: float = 0.5, random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__()
        self.p = p
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter: Augmenter = iaa.Sometimes(self.p, iaa.Cutout(nb_iterations=(1, 10), size=(0.1, 0.5),
                                                                     random_state=self.random_state))
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        p = UniformFloatHyperparameter('p', lower=0.2, upper=1, default_value=0.5)
        cs.add_hyperparameters([p])
        return cs