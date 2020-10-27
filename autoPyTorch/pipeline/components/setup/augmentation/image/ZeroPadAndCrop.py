from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter
)

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class ZeroPadAndCrop(BaseImageAugmenter):
    def __init__(self, x_position: float = 0, y_position: float = 0,
                 percent: float = 0.1, random_state: Optional[Union[int, np.random.RandomState]] = None):
        super().__init__()
        self.random_state = random_state
        self.percent = percent
        self.position = (x_position, y_position)
        self.pad_augmenter: Optional[Augmenter] = None
        self.crop_augmenter: Optional[Augmenter] = None

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.check_requirements(X, y)
        self.pad_augmenter = iaa.Pad(percent=self.percent, keep_size=False)
        self.crop_augmenter = iaa.CropToFixedSize(height=X['image_height'], width=X['image_width'],
                                                  position=self.position)
        self.augmenter: Augmenter = iaa.Sequential([
            self.pad_augmenter,
            self.crop_augmenter
        ], name=self.get_properties()['name'])

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
        x_position = UniformFloatHyperparameter('x_position', lower=0, upper=1, default_value=0)
        y_position = UniformFloatHyperparameter('y_position', lower=0, upper=1, default_value=0)
        percent = UniformFloatHyperparameter('percent', lower=0, upper=0.5, default_value=0.1)
        cs.add_hyperparameters([x_position, y_position, percent])
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        return {'name': 'ZeroPadAndCrop'}
