from typing import Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from imgaug.augmenters.meta import Augmenter

import numpy as np

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class BaseImageAugmenter(autoPyTorchSetupComponent):
    def __init__(self) -> None:
        super().__init__()
        self.augmenter: Optional[Augmenter] = None

    def get_image_augmenter(self) -> Augmenter:
        """
        Get fitted augmenter. Can only be called if fit()
        has been called on the object.
        Returns:
            BaseEstimator: Fitted augmentor
        """
        if self.augmenter is None:
            raise AttributeError("Can't return augmenter for {}, as it has not been"
                                 " fitted yet".format(self.__class__.__name__))
        return self.augmenter
    #
    # def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
    #     """
    #     A mechanism in code to ensure the correctness of the fit dictionary
    #     It recursively makes sure that the children and parent level requirements
    #     are honored before fit.
    #
    #     Args:
    #         X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
    #             mechanism, in which during a transform, a components adds relevant information
    #             so that further stages can be properly fitted
    #     """
    #     super().check_requirements(X, y)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.augmenter is None:
            raise ValueError("cant call {} without fitting first."
                             .format(self.__class__.__name__))
        return self.augmenter(images=X)

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
