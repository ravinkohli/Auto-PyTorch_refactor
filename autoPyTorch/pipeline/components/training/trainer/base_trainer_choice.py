import collections
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
)

import numpy as np

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.training.losses import get_loss_instance
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.trainer.base_trainer import (
    BaseTrainerComponent,
    BudgetTracker,
    RunSummary,
)
from autoPyTorch.utils import logging_ as logging


directory = os.path.split(__file__)[0]
_trainers = find_components(__package__,
                            directory,
                            BaseTrainerComponent)
_addons = ThirdPartyComponents(BaseTrainerComponent)


def add_trainer(trainer: BaseTrainerComponent) -> None:
    _addons.add_component(trainer)


class TrainerChoice(autoPyTorchChoice):
    """This class is an interface to the PyTorch trainer.


    To map to pipeline terminology, a choice component will implement the epoch
    loop through fit, whereas the component who is chosen will dictate how a single
    epoch happens, that is, how batches of data are fed and used to train the network.

    """
    def __init__(self,
                 dataset_properties: Dict[str, Any],
                 random_state: Optional[np.random.RandomState] = None
                 ):

        super().__init__(dataset_properties=dataset_properties,
                         random_state=random_state)
        self.run_summary = None  # Optional[RunSummary]
        self.writer = None  # Optional[SummaryWriter]

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available trainer components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all components available
                as choices for learning rate scheduling
        """
        components = collections.OrderedDict()
        components.update(_trainers)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(
        self,
        dataset_properties: Optional[Dict[str, str]] = None,
        default: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> ConfigurationSpace:
        """Returns the configuration space of the current chosen components

        Args:
            dataset_properties (Optional[Dict[str, str]]): Describes the dataset to work on
            default (Optional[str]): Default scheduler to use
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                 chosen component
        """
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal preprocessors for this problem
        available_trainers = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        if len(available_trainers) == 0:
            raise ValueError("No trainer found")

        if default is None:
            defaults = ['StandartTrainer',
                        ]
            for default_ in defaults:
                if default_ in available_trainers:
                    default = default_
                    break

        trainer = CategoricalHyperparameter(
            '__choice__',
            list(available_trainers.keys()),
            default_value=default
        )
        cs.add_hyperparameter(trainer)
        for name in available_trainers:
            trainer_configuration_space = available_trainers[name]. \
                get_hyperparameter_search_space(dataset_properties)
            parent_hyperparameter = {'parent': trainer, 'value': name}
            cs.add_configuration_space(
                name,
                trainer_configuration_space,
                parent_hyperparameter=parent_hyperparameter
            )

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        X.update({'run_summary': self.run_summary})
        return X

    def fit(self, X: Dict[str, Any], y: Any = None, **kwargs: Any) -> autoPyTorchComponent:
        """
        Fits a component by using an input dictionary with pre-requisites

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API

        Returns:
            A instance of self
        """

        # Comply with mypy
        assert self.choice is not None

        # Make sure that the prerequisites are there
        self.check_requirements(X, y)

        # Setup a Logger and other logging support
        logger = logging.get_logger(X['job_id'])
        if 'use_tensorboard_logger' in X and X['use_tensorboard_logger']:
            self.writer = SummaryWriter(log_dir=X['working_dir'])

        if X["torch_num_threads"] > 0:
            torch.set_num_threads(X["torch_num_threads"])

        budget_tracker = BudgetTracker(
            budget_type=X['budget_type'],
            max_value=X['runtime'] if X['budget_type'] == 'runtime' else X['epochs'],
        )

        self.choice.prepare(
            model=X['network'],
            metrics=[m() for m in get_metrics(X['dataset_properties'])],
            criterion=get_loss_instance(X['dataset_properties']),
            budget_tracker=budget_tracker,
            optimizer=X['optimizer'],
            device=self.get_device(X),
            logger=logger,
            writer=self.writer,
        )
        total_parameter_count, trainable_parameter_count = self.count_parameters(X['network'])
        self.run_summary = RunSummary(
            total_parameter_count,
            trainable_parameter_count,
        )

        epoch = 1

        while True:

            # prepare epoch
            start_time = time.time()

            self.choice.on_epoch_start(X=X, epoch=epoch)

            # training
            train_loss, train_metrics = self.choice.train(
                train_loader=X['train_data_loader'],
                epoch=epoch,
            )

            val_loss, val_metrics, test_loss, test_metrics = None, {}, None, {}
            if self.eval_valid_each_epoch(X):
                val_loss, val_metrics = self.choice.evaluate(X['val_data_loader'], epoch)
                if 'test_data_loader' in X and X['test_data_loader']:
                    test_loss, test_metrics = self.choice.evaluate(X['test_data_loader'], epoch)

            # TODO: does it make sense to call all schedulers here?
            # Some people exhaust the learning rate on every epoch
            # others on a batch basis. Ask!
            if X['lr_scheduler']:
                if 'ReduceLROnPlateau' in X['lr_scheduler'].__class__.__name__:
                    X['lr_scheduler'].step(val_loss)
                else:
                    X['lr_scheduler'].step()

            # Save training information
            self.run_summary.add_performance(
                epoch=epoch,
                start_time=start_time,
                end_time=time.time(),
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )

            # Save the weights of the best model and, if patience
            # exhausted break training
            if self.early_stop_handler(X):
                break

            if self.choice.on_epoch_end(X=X, epoch=epoch):
                break

            logger.debug(self.run_summary.repr_last_epoch())

            # Reached max epoch on next iter, don't even go there
            if budget_tracker.is_max_epoch_reached(epoch + 1):
                break

            epoch += 1

            torch.cuda.empty_cache()

        # wrap up -- add score if not evaluating every epoch
        if not self.eval_valid_each_epoch(X):
            val_loss, val_metrics = self.choice.evaluate(X['val_data_loader'])
            if 'test_data_loader' in X and X['val_data_loader']:
                test_loss, test_metrics = self.choice.evaluate(X['test_data_loader'])
            self.run_summary.add_performance(
                epoch=epoch,
                start_time=start_time,
                end_time=time.time(),
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
            )
            logger.debug(self.run_summary.repr_last_epoch())
            self.save_model_for_ensemble()

        logger.info(f"Finished training with {self.run_summary.repr_last_epoch()}")

        # Tag as fitted
        self.fitted_ = True

        return self

    def early_stop_handler(self, X: Dict[str, Any]) -> bool:
        """
        If early stopping is enabled, this procedure stops the training after a
        given patience
        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted

        Returns:
            bool: If true, training should be stopped
        """
        assert self.run_summary is not None
        bad_epochs = self.run_summary.get_best_epoch() - self.run_summary.get_last_epoch()
        if bad_epochs > X['early_stopping']:
            return True

        return False

    def eval_valid_each_epoch(self, X: Dict[str, Any]) -> bool:
        """
        Returns true if we are supposed to evaluate the model on every epoch,
        on the validation data. Usually, we only validate the data at the end,
        but in the case of early stopping, is appealing to evaluate each epoch.
        Args:
            X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted

        Returns:
            bool: if True, the model is evaluated in every epoch

        """
        if 'early_stopping' in X and X['early_stopping']:
            return True

        # We need to know if we should reduce the rate based on val loss
        if 'ReduceLROnPlateau' in X['lr_scheduler'].__class__.__name__:
            return True

        return False

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

        # We need a working dir in where to put our data
        if 'working_dir' not in X:
            raise ValueError('Need a working directory to output trainer information, '
                             "yet 'working_dir' was not found in the fit dictionary")

        # Setup Components
        if 'lr_scheduler' not in X:
            raise ValueError("Learning rate scheduler not found in the fit dictionary!")

        if 'network' not in X:
            raise ValueError("Network not found in the fit dictionary!")

        if 'optimizer' not in X:
            raise ValueError("Optimizer not found in the fit dictionary!")

        # Training Components
        if 'train_data_loader' not in X:
            raise ValueError("train_data_loader not found in the fit dictionary!")

        if 'val_data_loader' not in X:
            raise ValueError("val_data_loader not found in the fit dictionary!")

        if 'budget_type' not in X:
            raise ValueError("Budget type not found in the fit dictionary!")
        else:
            if 'epochs' not in X or 'runtime' not in X:
                if X['budget_type'] == 'epochs' and 'epochs' not in X:
                    raise ValueError("Budget type is epochs but "
                                     "no epochs was not found in the fit dictionary!")
                elif X['budget_type'] == 'runtime' and 'runtime' not in X:
                    raise ValueError("Budget type is runtime but "
                                     "no maximum number of seconds was provided!")
            else:
                raise ValueError("Unsupported budget type provided: {}".format(
                    X['budget_type']
                ))

        if 'job_id' not in X:
            raise ValueError('Need a job identifier to be able to isolate jobs')

        for config_option in ["torch_num_threads", 'device']:
            if config_option not in X:
                raise ValueError("Missing config option {} in config".format(
                    config_option
                ))

    def get_device(self, X: Dict[str, Any]) -> torch.device:
        """
        Returns the device to do torch operations

        Args:
            X (Dict[str, Any]): A fit dictionary to control how the pipeline
                is fitted

        Returns:
            torch.device: the device in which to compute operations. Cuda/cpu
        """
        if not torch.cuda.is_available():
            return torch.device('cpu')
        return torch.device(X['device'])

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
        """
        A method to get the total/trainable parameter count from the model

        Args:
            model (torch.nn.Module): the module from which to count parameters

        Returns:
            total_parameter_count: the total number of parameters of the model
            trainable_parameter_count: only the parameters being optimized
        """
        total_parameter_count = sum(
            p.numel() for p in model.parameters())
        trainable_parameter_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        return total_parameter_count, trainable_parameter_count

    def save_model_for_ensemble(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = str(self.run_summary)
        return string
