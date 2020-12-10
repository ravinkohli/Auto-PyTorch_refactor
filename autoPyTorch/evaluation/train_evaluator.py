from multiprocessing.queues import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration
import numpy as np

import pandas as pd

from smac.tae import StatusType

from sklearn.base import BaseEstimator

from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    MULTICLASSMULTIOUTPUT,
)
from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    fit_and_suppress_warnings
)
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.utils.backend import Backend


__all__ = ['TrainEvaluator', 'eval_holdout']


def _get_y_array(y: np.ndarray, task_type: int) -> np.ndarray:
    if task_type in CLASSIFICATION_TASKS and task_type != \
            MULTICLASSMULTIOUTPUT:
        return y.ravel()
    else:
        return y


class TrainEvaluator(AbstractEvaluator):
    def __init__(self, backend: Backend, queue: Queue, metric: List[autoPyTorchMetric],
                 configuration: Optional[Configuration] = None,
                 seed: int = 1,
                 output_y_hat_optimization: bool = True,
                 num_run: Optional[int] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 disable_file_output: bool = False,
                 init_params: Optional[Dict[str, Any]] = None,
                 budget: Optional[float] = None,
                 budget_type: Optional[str] = None,
                 logger_port: Optional[int] = None,
                 keep_models: Optional[bool] = None) -> None:
        super().__init__(
            backend=backend,
            queue=queue,
            configuration=configuration,
            metric=metric,
            seed=seed,
            output_y_hat_optimization=output_y_hat_optimization,
            num_run=num_run,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output,
            init_params=init_params,
            budget=budget,
            budget_type=budget_type,
            logger_port=logger_port
        )

        self.splits = self.datamanager.splits
        if self.splits is None:
            raise AttributeError("Must have called create_splits on {}".format(self.datamanager.__class__.__name__))
        self.num_folds = len(self.splits)
        self.Y_optimization = None
        self.Y_targets = [None] * self.num_folds
        self.Y_train_targets = np.ones(self.y_train.shape) * np.NaN
        self.models = [None] * self.num_folds
        self.indices = [None] * self.num_folds  # type: List[Optional[Tuple[Union[np.ndarray, List]]]]

        self.keep_models = keep_models

    def fit_predict_and_loss(self) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout"""
        assert self.splits is not None, "Can't fit pipeline in {} is datamanager.splits is None"\
            .format(self.__class__.__name__)
        for i, (train_split, test_split) in enumerate(self.splits):
            self.logger.info("Starting fit {}".format(i))

            self.Y_optimization = self.y_train[test_split]
            self.Y_actual_train = self.y_train[train_split]
            self._fit_and_predict(i, train_indices=train_split,
                                  test_indices=test_split,
                                  add_model_to_self=True)

    def _fit_and_predict(self, fold: int, train_indices: Union[np.ndarray, List], test_indices: Union[np.ndarray, List],
                         add_model_to_self: bool) -> None:

        model = self._get_model()

        self.indices[fold] = ((train_indices, test_indices))

        # Do only output the files in the case of iterative holdout,
        # In case of iterative partial cv, no file output is needed
        # because ensembles cannot be built
        file_output = True if self.num_folds == 1 else False
        X = {'train_indices': train_indices,
             'val_indices': test_indices,
             'split_id': fold,
             **self._init_params}  # fit dictionary
        y = None
        fit_and_suppress_warnings(self.logger, model, X, y)
        self.logger.info("Model fitted, now predicting")
        (
            Y_train_pred,
            Y_valid_pred,
            Y_test_pred
        ) = self._predict(
            model,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        if add_model_to_self:
            self.model = model

        train_loss = self._loss(self.y_train[train_indices], Y_train_pred)
        loss = self._loss(self.y_train[test_indices], Y_valid_pred)
        additional_run_info = model.get_additional_run_info()

        status = StatusType.SUCCESS

        self.finish_up(
            loss=loss,
            train_loss=train_loss,
            # TODO: currently sending valid_pred to all,
            #  potentially change valid_pred to opt_pred
            opt_pred=Y_valid_pred,
            valid_pred=Y_valid_pred,
            val_indices=test_indices,
            test_pred=Y_test_pred,
            additional_run_info=additional_run_info,
            file_output=file_output,
            status=status,
        )

    def _predict(self, model: BaseEstimator,
                 test_indices: Union[np.ndarray, List],
                 train_indices: Union[np.ndarray, List]):

        X_train = self.X_train
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        train_pred = self.predict_function(X_train[train_indices],
                                           model, self.task_type,
                                           self.y_train[train_indices])

        valid_pred = self.predict_function(X_train[test_indices],
                                           model, self.task_type,
                                           self.y_train[train_indices])

        if self.X_test is not None:
            test_pred = self.predict_function(self.X_test, model,
                                              self.task_type,
                                              self.y_train[train_indices])
        else:
            test_pred = None

        return train_pred, valid_pred, test_pred


# create closure for evaluating an algorithm
def eval_holdout(
        queue,
        config,
        backend,
        metric,
        seed,
        num_run,
        instance,
        output_y_hat_optimization,
        include,
        exclude,
        disable_file_output,
        init_params=None,
        budget=100.0,
        budget_type=None,
        logger_port=None
):
    evaluator = TrainEvaluator(
        backend=backend,
        queue=queue,
        metric=metric,
        configuration=config,
        seed=seed,
        num_run=num_run,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
        logger_port=logger_port
    )
    evaluator.fit_predict_and_loss()
