from typing import Any, Dict, Optional, Union

import numpy as np

import torch.tensor

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.padding.base_pad import (
    BasePad
)


class NoPad(BasePad):

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        self.random_state = random_state

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
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, str]] = None
                       ) -> Dict[str, Any]:
        return {
            'shortname': 'pad',
            'name': 'Padding Node',
        }
