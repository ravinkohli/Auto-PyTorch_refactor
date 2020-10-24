from typing import Any, Dict, Optional

import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter

from autoPyTorch.pipeline.components.setup.augmentation.image.base_image_augmenter import BaseImageAugmenter


class HorizontalFlip(BaseImageAugmenter):
    def __init__(self, random_state: Optional[int, np.random.RandomState] = None):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseImageAugmenter:
        self.augmenter: Augmenter = iaa.Fliplr(p=0.5)

        return self
