import time
from typing import Dict, Optional
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor

from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    MULTICLASS,
    TABULAR_TASKS,
    IMAGE_TASKS
)
from autoPyTorch.evaluation.utils import (
    convert_multioutput_multiclass_to_multilabel
)
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score, CLASSIFICATION_METRICS, REGRESSION_METRICS
from autoPyTorch.utils.backend import Backend
from autoPyTorch.utils.logging_ import get_named_client_logger

from ConfigSpace import Configuration


__all__ = [
    'AbstractEvaluator'
]

# TODO: make dummypipeline or model for autopytorch
class MyDummyClassifier(DummyClassifier):
    def __init__(self, config, random_state, init_params=None):
        self.configuration = config
        if config == 1:
            super(MyDummyClassifier, self).__init__(strategy="uniform")
        else:
            super(MyDummyClassifier, self).__init__(strategy="most_frequent")

    def pre_transform(self, X, y, fit_params=None):  # pylint: disable=R0201
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X, y, sample_weight=None):
        X_train = X['X_train'][X['train_indices']]
        y_train = X['y_train'][X['train_indices']]
        return super(MyDummyClassifier, self).fit(np.ones((X_train.shape[0], 1)), y_train,
                                                  sample_weight=sample_weight)

    def predict_proba(self, X, batch_size=1000):
        new_X = np.ones((X['X_train']['val_indices'].shape[0], 1))
        probas = super(MyDummyClassifier, self).predict_proba(new_X)
        probas = convert_multioutput_multiclass_to_multilabel(probas).astype(
            np.float32)
        return probas

    def estimator_supports_iterative_fit(self):  # pylint: disable=R0201
        return False

    def get_additional_run_info(self):  # pylint: disable=R0201
        return None


class MyDummyRegressor(DummyRegressor):
    def __init__(self, config, random_state, init_params=None):
        self.configuration = config
        if config == 1:
            super(MyDummyRegressor, self).__init__(strategy='mean')
        else:
            super(MyDummyRegressor, self).__init__(strategy='median')

    def pre_transform(self, X, y, fit_params=None):
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X, y, sample_weight=None):
        X_train = X['X_train'][X['train_indices']]
        y_train = X['y_train'][X['train_indices']]
        return super(MyDummyRegressor, self).fit(np.ones((X_train.shape[0], 1)), y_train,
                                                 sample_weight=sample_weight)

    def predict(self, X, batch_size=1000):
        new_X = np.ones((X['X_train']['val_indices'].shape[0], 1))
        return super(MyDummyRegressor, self).predict(new_X).astype(np.float32)

    def estimator_supports_iterative_fit(self):  # pylint: disable=R0201
        return False

    def get_additional_run_info(self):  # pylint: disable=R0201
        return None


def _fit_and_suppress_warnings(logger, model, X, y):
    def send_warnings_to_log(message, category, filename, lineno,
                             file=None, line=None):
        logger.debug('%s:%s: %s:%s',
                     filename, lineno, category.__name__, message)
        return

    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        model.fit(X, y)

    return model


# TODO: adjust for differences in image and tabular data, also maybe time series but its low priority
class AbstractEvaluator(object):
    def __init__(self, backend: Backend, queue, metric,
                 configuration=None,
                 seed=1,
                 output_y_hat_optimization=True,
                 num_run=None,
                 include=None,
                 exclude=None,
                 disable_file_output=False,
                 init_params=None,
                 budget=None,
                 budget_type=None,
                 logger_port=None):

        self.starttime = time.time()

        self.configuration = configuration
        # TODO: adjust backend for autopytorch backend
        self.backend = backend
        self.queue = queue

        # TODO: use autopytorch datasets once PR is passed
        self.datamanager = self.backend.load_datamanager()
        self.include = include
        self.exclude = exclude

        self.X_train = self.datamanager.data['X_train']
        self.y_train = self.datamanager.data['Y_train']

        self.X_test = self.datamanager.data.get('X_test')
        self.y_test = self.datamanager.data.get('Y_test')

        self.metric = metric
        self.task_type = self.datamanager.info['task_type']
        self.output_type = self.datamanager.info['output_type']
        self.seed = seed

        self.output_y_hat_optimization = output_y_hat_optimization
        # TODO: Check if we need all supported metrics, as in our case even single metric is in a score_dict form

        if isinstance(disable_file_output, (bool, list)):
            self.disable_file_output = disable_file_output
        else:
            raise ValueError('disable_file_output should be either a bool or a list')

        if self.task_type in REGRESSION_TASKS:
            if not isinstance(self.configuration, Configuration):
                self.model_class = MyDummyRegressor
            else:
                if self.task_type in TABULAR_TASKS:
                    self.model_class = TabularRegressionPipeline
                else:
                    raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_regression
        else:
            if not isinstance(self.configuration, Configuration):
                self.model_class = MyDummyClassifier
            else:
                if self.task_type in TABULAR_TASKS:
                    self.model_class = TabularClassificationPipeline
                elif self.task_type in IMAGE_TASKS:
                    self.model_class = ImageClassificationPipeline
                else:
                    raise ValueError('task {} not available'.format(self.task_type))
            self.predict_function = self._predict_proba
        if self.task_type in TABULAR_TASKS:
            categorical_columns = []
            numerical_columns = []
            for i, feat in enumerate(self.datamanager.feat_type):
                if feat.lower() == 'numerical':
                    numerical_columns.append(i)
                elif feat.lower() == 'categorical':
                    categorical_columns.append(i)
                else:
                    raise ValueError(feat)

            categories = [np.unique(self.X_train[a]).tolist() for a in categorical_columns]
            # TODO: maybe change to fit_dictionary
            self._init_params = {
                'categorical_columns':
                    categorical_columns,
                'numerical_columns':
                    numerical_columns,
                'categories':
                    categories
            }
            self.dataset_properties = {
                'categorical_columns':
                    categorical_columns,
                'numerical_columns':
                    numerical_columns,}
            if init_params is not None:
                self._init_params.update(init_params)
        elif self.task_type in IMAGE_TASKS:
            # TODO: Add image dataset properties to data manager
            self._init_params = {
                'image_height':
                    self.datamanager.info['image_height'],
                'image_width':
                    self.datamanager.info['image_width']
            }
        # Create dataset_properties
        # TODO: Change for our pipeline
        self.dataset_properties.update({
                'task_type': self.task_type,
                'sparse': self.datamanager.info['is_sparse'] == 1,
                'output_type': self.output_type
            })

        self._init_params.update({
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'dataset_properties': self.dataset_properties,
        })

        if num_run is None:
            num_run = 0
        self.num_run = num_run

        logger_name = '%s(%d):%s' % (self.__class__.__name__.split('.')[-1],
                                     self.seed, self.datamanager.name)
        self.logger = get_named_client_logger(name=logger_name, port=logger_port,
                                              output_dir=self.backend.temporary_directory)

        self.Y_optimization = None
        self.Y_actual_train = None

        self.budget = budget
        self.budget_type = budget_type
        default_pipeline_options = self.model_class.get_default_pipeline_options()
        if self.budget_type == 'runtime':
            default_pipeline_options['runtime'] = self.budget
            if 'epochs' in default_pipeline_options:
                del default_pipeline_options['epochs']
        elif self.budget_type == 'epochs':
            default_pipeline_options['epochs'] = self.budget
            if 'runtime' in default_pipeline_options:
                del default_pipeline_options['runtime']
        default_pipeline_options['budget_type'] = self.budget_type
        self._init_params.update({**default_pipeline_options})

    def _get_model(self):
        if not isinstance(self.configuration, Configuration):
            model = self.model_class(config=self.configuration,
                                     random_state=np.random.RandomState(self.seed),
                                     init_params=self._init_params)
        else:

            model = self.model_class(config=self.configuration,
                                     dataset_properties=self.dataset_properties,
                                     random_state=np.random.RandomState(self.seed),
                                     include=self.include,
                                     exclude=self.exclude,
                                     init_params=self._init_params)
        return model

    def _loss(self, y_true, y_hat, all_supported_metrics=None):
        """Auto-sklearn follows a minimization goal, so the make_scorer
        sign is used as a guide to obtain the value to reduce.

        On this regard, to optimize a metric:
            1- score is calculared with calculate_score, with the caveat, that if
            for the metric greater is not better, a negative score is returned.
            2- the err (the optimization goal) is then:
                optimum - (metric.sign * actual_score)
                For accuracy for example: optimum(1) - (+1 * actual score)
                For logloss for example: optimum(0) - (-1 * actual score)
        """
        # TODO: removed all_supported_metrics

        if not isinstance(self.configuration, Configuration):
            return {self.metric: 1.0}

        score = calculate_score(
            y_true, y_hat, self.task_type, self.metric)

        if hasattr(score, '__len__'):
            if self.task_type in CLASSIFICATION_TASKS:
                err = {key: metric._optimum - score[key] for key, metric in
                       CLASSIFICATION_METRICS.items() if key in score}
            else:
                err = {key: metric._optimum - score[key] for key, metric in
                       REGRESSION_METRICS.items() if key in score}
        else:
            err = self.metric._optimum - score

        return err

    def finish_up(self, loss, train_loss,  opt_pred, valid_pred, test_pred,
                  additional_run_info, file_output, status):
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        self.duration = time.time() - self.starttime

        if file_output:
            loss_, additional_run_info_ = self.file_output(
                opt_pred, valid_pred, test_pred,
            )
        else:
            loss_ = None
            additional_run_info_ = {}

        validation_loss, test_loss = self.calculate_auxiliary_losses(
            valid_pred, test_pred,
        )

        if loss_ is not None:
            return self.duration, loss_, self.seed, additional_run_info_

        if isinstance(loss, dict):
            loss_ = loss
            loss = loss_[self.metric.name]
        else:
            loss_ = {}

        additional_run_info = (
            {} if additional_run_info is None else additional_run_info
        )
        for metric_name, value in loss_.items():
            additional_run_info[metric_name] = value
        additional_run_info['duration'] = self.duration
        additional_run_info['num_run'] = self.num_run
        if train_loss is not None:
            additional_run_info['train_loss'] = train_loss
        if validation_loss is not None:
            additional_run_info['validation_loss'] = validation_loss
        if test_loss is not None:
            additional_run_info['test_loss'] = test_loss

        rval_dict = {'loss': loss,
                     'additional_run_info': additional_run_info,
                     'status': status}
        # if final_call:
        #     rval_dict['final_queue_element'] = True

        self.queue.put(rval_dict)

    def calculate_auxiliary_losses(
        self,
        Y_valid_pred,
        Y_test_pred
    ):
        if Y_valid_pred is not None:
            if self.y_valid is not None:
                validation_loss = self._loss(self.y_valid, Y_valid_pred)
                if isinstance(validation_loss, dict):
                    validation_loss = validation_loss[self.metric.name]
            else:
                validation_loss = None
        else:
            validation_loss = None

        if Y_test_pred is not None:
            if self.y_test is not None:
                test_loss = self._loss(self.y_test, Y_test_pred)
                if isinstance(test_loss, dict):
                    test_loss = test_loss[self.metric.name]
            else:
                test_loss = None
        else:
            test_loss = None

        return validation_loss, test_loss

    def file_output(
            self,
            Y_optimization_pred,
            Y_valid_pred,
            Y_test_pred
    ):
        # Abort if self.Y_optimization is None
        # self.Y_optimization can be None if we use partial-cv, then,
        # obviously no output should be saved.
        if self.Y_optimization is None:
            return None, {}

        # Abort in case of shape misalignment
        if self.Y_optimization.shape[0] != Y_optimization_pred.shape[0]:
            return (
                1.0,
                {
                    'error':
                        "Targets %s and prediction %s don't have "
                        "the same length. Probably training didn't "
                        "finish" % (self.Y_optimization.shape, Y_optimization_pred.shape)
                 },
            )

        # Abort if predictions contain NaNs
        for y, s in [
            # Y_train_pred deleted here. Fix unittest accordingly.
            [Y_optimization_pred, 'optimization'],
            [Y_valid_pred, 'validation'],
            [Y_test_pred, 'test']
        ]:
            if y is not None and not np.all(np.isfinite(y)):
                return (
                    1.0,
                    {
                        'error':
                            'Model predictions for %s set contains NaNs.' % s
                    },
                )

        # Abort if we don't want to output anything.
        # Since disable_file_output can also be a list, we have to explicitly
        # compare it with True.
        if self.disable_file_output is True:
            return None, {}

        # Notice that disable_file_output==False and disable_file_output==[]
        # means the same thing here.
        if self.disable_file_output is False:
            self.disable_file_output = []

        # This file can be written independently of the others down bellow
        if ('y_optimization' not in self.disable_file_output):
            if self.output_y_hat_optimization:
                self.backend.save_targets_ensemble(self.Y_optimization)

        if hasattr(self, 'models') and len(self.models) > 0 and self.models[0] is not None:
            if ('models' not in self.disable_file_output):

                if self.task_type in CLASSIFICATION_TASKS:
                    models = VotingClassifier(estimators=None, voting='soft', )
                else:
                    models = VotingRegressor(estimators=None)
                models.estimators_ = self.models
            else:
                models = None
        else:
            models = None

        self.backend.save_numrun_to_dir(
            seed=self.seed,
            idx=self.num_run,
            budget=self.budget,
            model=self.model if 'model' not in self.disable_file_output else None,
            cv_model=models if 'cv_model' not in self.disable_file_output else None,
            ensemble_predictions=(
                Y_optimization_pred if 'y_optimization' not in self.disable_file_output else None
            ),
            valid_predictions=(
                Y_valid_pred if 'y_valid' not in self.disable_file_output else None
            ),
            test_predictions=(
                Y_test_pred if 'y_test' not in self.disable_file_output else None
            ),
        )

        return None, {}

    def _predict_proba(self, X, model, task_type, Y_train):
        def send_warnings_to_log(message, category, filename, lineno,
                                 file=None, line=None):
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict_proba(X, batch_size=1000)

        Y_pred = self._ensure_prediction_array_sizes(Y_pred, Y_train)
        return Y_pred

    def _predict_regression(self, X, model, task_type, Y_train=None):
        def send_warnings_to_log(message, category, filename, lineno,
                                 file=None, line=None):
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, category.__name__, message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict(X, batch_size=1000)

        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))

        return Y_pred

    def _ensure_prediction_array_sizes(self, prediction, Y_train):
        num_classes = self.datamanager.info['label_num']

        if self.output_type == MULTICLASS and \
                prediction.shape[1] < num_classes:
            if Y_train is None:
                raise ValueError('Y_train must not be None!')
            classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros((prediction.shape[0], num_classes),
                                       dtype=np.float32)

            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction
