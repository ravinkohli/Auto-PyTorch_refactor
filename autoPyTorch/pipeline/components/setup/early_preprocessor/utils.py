from typing import Any, Dict, List

import numpy as np

from sklearn.utils import check_array

import torchvision.transforms

from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import autoPyTorchPreprocessingComponent


def get_preprocess_transforms(X: Dict[str, Any]) -> torchvision.transforms.Compose:
    transforms = list()  # type: List[autoPyTorchPreprocessingComponent]
    for key, value in X.items():
        if isinstance(value, autoPyTorchPreprocessingComponent):
            transforms.append(value)

    return torchvision.transforms.Compose(transforms)


def preprocess(dataset: np.ndarray, transforms: torchvision.transforms.Compose,
               indices: List[int] = None) -> np.ndarray:

    # In case of pandas dataframe, make sure we comply with sklearn API,
    # also, we require numpy for the next transformations
    # We use the same query for iloc as sklearn uses in its estimators
    if hasattr(dataset, 'iloc'):
        dataset = check_array(dataset)

    if indices is None:
        dataset = transforms(dataset)
    else:
        dataset[indices, :] = transforms(np.take(dataset, indices, axis=0))
    return dataset
