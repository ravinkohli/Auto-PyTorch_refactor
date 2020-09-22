from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import torch.tensor

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.padding.base_pad import BasePad


class Pad(BasePad):
    def __init__(self, border: int = 2,
                 mode: str = 'constant',
                 random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        self.random_state = random_state
        self.border = border
        self.mode = mode

    def __call__(self, X: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        """
        Makes the autoPyTorchPreprocessingComponent Callable. Calling the component
        calls the transform function of the underlying preprocessor and
        returns the transformed array.
        Args:
            X (Union[np.ndarray, torch.tensor]): input data tensor

        Returns:
            Union[np.ndarray, torch.tensor]: Transformed data tensor
        """
        return np.pad(X, [(0, 0), (self.border, self.border), (self.border, self.border), (0, 0)], mode=self.mode)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        border = UniformIntegerHyperparameter('border', lower=1, upper=8, default_value=2)
        mode = CategoricalHyperparameter('mode',
                                         choices=['constant', 'edge', 'maximum', 'mean',
                                                  'median', 'minimum', 'reflect', 'wrap'],
                                         default_value='constant')
        cs.add_hyperparameters([border, mode])

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        return {
            'shortname': 'pad',
            'name': 'Padding Node',
        }
