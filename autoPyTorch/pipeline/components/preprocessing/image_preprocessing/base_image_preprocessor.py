from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

import torch

from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import autoPyTorchPreprocessingComponent


class autoPyTorchImagePreprocessingComponent(autoPyTorchPreprocessingComponent):
    """
     Provides abstract interface for preprocessing algorithms in AutoPyTorch.
    """

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> "autoPyTorchImagePreprocessingComponent":
        """
        Initialises preprocessor and returns self.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            autoPyTorchImagePreprocessingComponent: self
        """
        self.check_requirements(X, y)

        return self
