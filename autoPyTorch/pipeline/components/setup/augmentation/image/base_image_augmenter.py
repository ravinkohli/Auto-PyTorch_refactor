from typing import Optional

from imgaug.augmenters.meta import Augmenter
from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent


class BaseImageAugmenter(autoPyTorchSetupComponent):
    def __init__(self):
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
