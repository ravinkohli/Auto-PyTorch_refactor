from typing import Dict, Optional, Union

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

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

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('preprocessor', None)
        info.pop('column_transformer')
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
