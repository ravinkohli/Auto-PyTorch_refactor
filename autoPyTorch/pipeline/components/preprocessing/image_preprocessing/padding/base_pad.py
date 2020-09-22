from typing import Any, Dict

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.base_image_preprocessor import \
    autoPyTorchImagePreprocessingComponent


class BasePad(autoPyTorchImagePreprocessingComponent):
    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:

        X.update({'pad': self})
        return X

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
