from typing import Any, Dict, Optional

import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.utils.common import FitRequirement


class BaseLRComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for schedulers
    in Auto-Pytorch"""

    _fit_requirements = [FitRequirement('optimizer', Optimizer)]

    def __init__(self) -> None:
        super().__init__()
        self.scheduler = None  # type: Optional[_LRScheduler]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'lr_scheduler': self.scheduler})
        return X

    def get_scheduler(self) -> _LRScheduler:
        """Return the underlying scheduler object.
        Returns:
            scheduler : the underlying scheduler object
        """
        assert self.scheduler is not None, "No scheduler was fit"
        return self.scheduler

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.scheduler.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('scheduler', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
