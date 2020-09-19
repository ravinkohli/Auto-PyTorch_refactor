from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

import numpy as np

import torch.nn
import torch.tensor

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.base_image_preprocessor import (
    autoPyTorchImagePreprocessingComponent

)


class Pad(autoPyTorchImagePreprocessingComponent):
    def __init__(self, padding: int):
        self.border = padding

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        X.update({'pad': self})
        return X

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
        return torch.nn.ReflectionPad2d(self.border)(X)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        padding = UniformIntegerHyperparameter('padding', lower=1, upper=8, default_value=2)
        cs.add_hyperparameter(padding)

        return cs