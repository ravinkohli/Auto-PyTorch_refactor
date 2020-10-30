from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.utils import check_array

import torch

import torchvision

from autoPyTorch.pipeline.components.training.base_training import autoPyTorchTrainingComponent
from autoPyTorch.pipeline.components.training.data_loader.transformable_tensor_dataset import (
    CustomXYTensorDataset
)


class BaseDataLoaderComponent(autoPyTorchTrainingComponent):
    """This class is an interface to the PyTorch Dataloader.

    It gives the possibility to read various types of mapped
    datasets as described in:
    https://pytorch.org/docs/stable/data.html

    """

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        self.train_dataset = None  # type: Optional[torch.utils.data.Dataset]
        self.val_dataset = None  # type: Optional[torch.utils.data.Dataset]
        self.train_data_loader = None  # type: Optional[torch.utils.data.DataLoader]
        self.val_data_loader = None  # type: Optional[torch.utils.data.DataLoader]

        # We also support existing datasets!
        self.dataset = None
        self.vision_datasets = self.get_torchvision_datasets()

        # Save the transformations for reuse
        self.train_transform = None  # type: Optional[torchvision.transforms.Compose]
        self.val_transform = None  # type: Optional[torchvision.transforms.Compose]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'train_data_loader': self.train_data_loader,
                  'val_data_loader': self.val_data_loader})
        return X

    def fit(self, X: Dict[str, Any], y: Any = None) -> torch.utils.data.DataLoader:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """

        # Make sure there is an optimizer
        self.check_requirements(X, y)

        self.train_transform = self.build_transform(X, train=True)
        self.val_transform = self.build_transform(X, train=False)

        if 'dataset' in X:
            self.train_dataset = self.get_torchvision_datasets()[X['dataset']](
                root=X['root'],
                transformation=self.train_transform,
                train=True,
            )
            self.val_dataset = self.get_torchvision_datasets()[X['dataset']](
                root=X['root'],
                transformation=self.val_transform,
                train=False,
            )
        else:
            # Make sure that the train data is numpy-compatible
            X_train = check_array(X['X_train'])
            y_train = check_array(X['y_train'], ensure_2d=False)
            self.train_dataset = CustomXYTensorDataset(
                X=np.take(X_train, X['train_indices'], axis=0),
                y=np.take(y_train, X['train_indices'], axis=0),
                transform=self.train_transform
            )
            self.val_dataset = CustomXYTensorDataset(
                X=np.take(X_train, X['val_indices'], axis=0),
                y=np.take(y_train, X['val_indices'], axis=0),
                transform=self.val_transform
            )

        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=min(self.batch_size, len(self.train_dataset)),
            shuffle=True,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', True),
        )

        self.val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=min(self.batch_size, len(self.val_dataset)),
            shuffle=False,
            num_workers=X.get('num_workers', 0),
            pin_memory=X.get('pin_memory', True),
            drop_last=X.get('drop_last', False),
        )

        return self

    def get_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int
                   ) -> torch.utils.data.DataLoader:
        """
        Creates a data loader object from the provided data,
        applying the transformations meant to validation objects
        """
        X = check_array(X)
        if y:
            y = check_array(y, ensure_2d=False)
        dataset = CustomXYTensorDataset(
            X=X,
            y=y,
            transform=self.val_transform
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
        )

    def build_transform(self, X: Dict[str, Any], train: bool = True) -> torchvision.transforms.Compose:
        """
        Method to build a transformation that can pre-process input data

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            train (bool)" whether transformation to be built are for training of test mode"

        Returns:
            A composition of transformations
        """
        raise NotImplementedError()

    def get_train_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the train data

        Returns:
            torch.utils.data.DataLoader: A train data loader
        """
        assert self.train_data_loader is not None, "No train data loader fitted"
        return self.train_data_loader

    def get_val_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns a data loader object for the validation data

        Returns:
            torch.utils.data.DataLoader: A validation data loader
        """
        assert self.val_data_loader is not None, "No val data loader fitted"
        return self.val_data_loader

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """

        # make sure the parent requirements are honored
        super().check_requirements(X, y)

        # We allow reading data from a user provided dataset
        # or from X, Y pairs
        if 'dataset' in X and 'X_train' in X:
            raise ValueError("Ambiguous definition of input for data loader. "
                             "Cannot provide both a dataset and a X,y input pair. "
                             "Currently X={}.".format(
                                 X
                             )
                             )
        elif 'dataset' not in X and 'X_train' not in X:
            raise ValueError("Data loader requires the user to provide the input data "
                             "via a dataset object or through X, y pairs but neither was "
                             "provided. Currently X={}.".format(
                                 X
                             )
                             )

        elif 'dataset' in X:

            if isinstance(X['dataset'], str) and X['dataset'] not in self.vision_datasets:
                raise ValueError(
                    "Unsupported dataset: {}. Supported datasets are {} ". format(
                        X['dataset'],
                        self.vision_datasets.keys(),
                    )
                )

            if 'root' not in X:
                raise ValueError("DataLoader needs the root of where the vision dataset will "
                                 "be located, yet X only contains {}.".format(
                                     X
                                 )
                                 )
        else:
            # We will be creating a tensor X,y dataset
            if 'X_train' not in X or 'y_train' not in X:
                raise ValueError("Data loader cannot access the train features-targets to "
                                 "be loaded. We expect both X_train and y_train to be arguments "
                                 "to the dataloader, yet X only contains {}.".format(
                                     X
                                 )
                                 )

            if 'train_indices' not in X or 'val_indices' not in X:
                raise ValueError("Data loader cannot access the indices needed to "
                                 "define training and validation data. "
                                 "X only contains {}.".format(
                                     X
                                 )
                                 )
        if 'is_small_preprocess' not in X:
            raise ValueError("is_small_pre-process is required to know if the data was preprocessed"
                             " or if the data-loader should transform it while loading a batch")

        # We expect this class to be a base for image/tabular/time
        # And the difference among this data types should be mainly
        # in the transform, so we delegate for special transformation checking
        # to the below method
        self._check_transform_requirements(X, y)

    def _check_transform_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """

        Makes sure that the fit dictionary contains the required transformations
        that the dataset should go through

        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        raise NotImplementedError()

    def get_torchvision_datasets(self) -> Dict[str, torchvision.datasets.VisionDataset]:
        """ Returns the supported dataset classes from torchvision

        This is gonna be used to instantiate a dataset object for the dataloader

        Returns:
            Dict[str, torchvision.datasets.VisionDataset]: A mapping from dataset name to class

        """
        return {
            'FashionMNIST': torchvision.datasets.FashionMNIST,
            'MNIST': torchvision.datasets.MNIST,
            'CIFAR10': torchvision.datasets.CIFAR10,
            'CIFAR100': torchvision.datasets.CIFAR100,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None
                                        ) -> ConfigurationSpace:
        batch_size = UniformIntegerHyperparameter(
            "batch_size", 32, 320, default_value=64)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([batch_size])
        return cs

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.train_data_loader.__class__.__name__
        info = vars(self)
        # Remove unwanted info
        info.pop('train_data_loader', None)
        info.pop('val_data_loader', None)
        info.pop('train_dataset', None)
        info.pop('val_dataset', None)
        info.pop('vision_datasets', None)
        info.pop('random_state', None)
        string += " (" + str(info) + ")"
        return string
