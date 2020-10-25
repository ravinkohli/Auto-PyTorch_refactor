from typing import Any, Dict

import torchvision

from autoPyTorch.pipeline.components.training.data_loader.base_data_loader import BaseDataLoaderComponent


class ImageDataLoader(BaseDataLoaderComponent):
    """This class is an interface to the PyTorch Dataloader.

    Particularly, this data loader builds transformations for
    image data.

    """

    def build_transform(self, X: Dict[str, Any], train: bool = True) -> torchvision.transforms.Compose:
        """
        Method to build a transformation that can pre-process input data

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            train (bool): whether transformation to be built are for training of test mode

        Returns:
            A composition of transformations
        """

        transformations = []

        if train:
            transformations.append(X['image_augmenter'])
        # In the case of image data, the options currently available
        # for preprocessors are:
        #   + normalise
        # These can apply for both train/val/test, so no
        # distinction is performed

        if not X['is_small_preprocess']:
            transformations.append(X['preprocess_transforms'])

        # Transform to tensor
        transformations.append(torchvision.transforms.ToTensor())

        return torchvision.transforms.Compose(transformations)

    def _check_transform_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """

        Makes sure that the fit dictionary contains the required transformations
        that the dataset should go through

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        if not X['image_augmenter'] and 'image_augmenter' not in X:
            raise ValueError("Cannot find the image_augmenter in the fit dictionary")

        if not X['is_small_preprocess'] and 'preprocess_transforms' not in X:
            raise ValueError("Cannot find the preprocess_transforms in the fit dictionary")
