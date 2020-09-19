from typing import Any, Dict, Optional, Union

import numpy as np

import torch.tensor

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.base_image_preprocessor import (
    autoPyTorchImagePreprocessingComponent

)


class NoPad(autoPyTorchImagePreprocessingComponent):

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
        return X
