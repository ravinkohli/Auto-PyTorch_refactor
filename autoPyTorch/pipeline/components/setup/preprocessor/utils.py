from typing import Any, Dict, List

import numpy as np

import torchvision.transforms

from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import autoPyTorchPreprocessingComponent


def get_preprocess_transforms(X: Dict[str, Any]) -> torchvision.transforms.Compose:
    transforms = list()  # type: List[autoPyTorchPreprocessingComponent]
    for key, value in X.items():
        if isinstance(value, autoPyTorchPreprocessingComponent):
            transforms.append(value)

    return torchvision.transforms.Compose(transforms)


def preprocess(dataset: np.ndarray, transforms: torchvision.transforms.Compose) -> np.ndarray:
    # uncomment next 2 lines if dataset is not np array
    # dataset = copy.copy(dataset)
    # dataset.data = transforms(dataset.data)

    dataset = transforms(dataset)
    return dataset
