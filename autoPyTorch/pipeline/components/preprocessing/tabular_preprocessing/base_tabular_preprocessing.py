from typing import Any, Dict, Optional, Union

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

import torch

from autoPyTorch.pipeline.components.preprocessing.base_preprocessing import autoPyTorchPreprocessingComponent


class autoPyTorchTabularPreprocessingComponent(autoPyTorchPreprocessingComponent):
    """
     Provides abstract interface for preprocessing algorithms in AutoPyTorch.
    """
    def __init__(self) -> None:
        self.preprocessor: Union[Dict[str, BaseEstimator], Optional[BaseEstimator]] = None
        self.column_transformer: Optional[ColumnTransformer] = None

    def get_column_transformer(self) -> ColumnTransformer:
        """
        Get fitted column transformer that is wrapped around
        the sklearn preprocessor. Can only be called if fit()
        has been called on the object.
        Returns:
            BaseEstimator: Fitted sklearn column transformer
        """
        return self.column_transformer

