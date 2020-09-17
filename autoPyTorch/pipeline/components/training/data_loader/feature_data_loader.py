from typing import Any, Dict

import torchvision

from autoPyTorch.pipeline.components.training.data_loader.base_data_loader import BaseDataLoaderComponent


class FeatureDataLoader(BaseDataLoaderComponent):
    """This class is an interface to the PyTorch Dataloader.

    Particularly, this data loader builds transformations for
    tabular data.

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

        # In the case of feature data, the options currently available
        # for transformations are:
        #   + imputer
        #   + encoder
        #   + scaler
        # This transformations apply for both train/val/test, so no
        # distinction is performed
        return torchvision.transforms.Compose([
            X['imputer'],
            X['scaler'],
            X['encoder'],
            torchvision.transforms.ToTensor(),
        ])

    def _check_transform_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """

        Makes sure that the fit dictionary contains the required transformations
        that the dataset should go through

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        for requirement in ['imputer', 'scaler', 'encoder']:
            if requirement not in X:
                raise ValueError("Cannot find the {} in the fit dictionary".format(
                    requirement
                ))
