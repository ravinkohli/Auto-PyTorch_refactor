import copy
import json

import numpy as np
from smac.tae import TAEAbortException, StatusType
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold, \
    StratifiedKFold, train_test_split, BaseCrossValidator, PredefinedSplit
from sklearn.model_selection._split import _RepeatedSplits, BaseShuffleSplit

from autoPyTorch.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    _fit_and_suppress_warnings,
)
from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    MULTICLASS,
    MULTICLASSMULTIOUTPUT,
    REGRESSION_TASKS,
    CONTINUOUSMULTIOUTPUT,
    STRING_TO_OUTPUT_TYPES
)


__all__ = ['TrainEvaluator', 'eval_holdout']

__baseCrossValidator_defaults__ = {'GroupKFold': {'n_splits': 3},
                                   'KFold': {'n_splits': 3,
                                             'shuffle': False,
                                             'random_state': None},
                                   'LeaveOneGroupOut': {},
                                   'LeavePGroupsOut': {'n_groups': 2},
                                   'LeaveOneOut': {},
                                   'LeavePOut': {'p': 2},
                                   'PredefinedSplit': {},
                                   'RepeatedKFold': {'n_splits': 5,
                                                     'n_repeats': 10,
                                                     'random_state': None},
                                   'RepeatedStratifiedKFold': {'n_splits': 5,
                                                               'n_repeats': 10,
                                                               'random_state': None},
                                   'StratifiedKFold': {'n_splits': 3,
                                                       'shuffle': False,
                                                       'random_state': None},
                                   'TimeSeriesSplit': {'n_splits': 3,
                                                       'max_train_size': None},
                                   'GroupShuffleSplit': {'n_splits': 5,
                                                         'test_size': None,
                                                         'random_state': None},
                                   'StratifiedShuffleSplit': {'n_splits': 10,
                                                              'test_size': None,
                                                              'random_state': None},
                                   'ShuffleSplit': {'n_splits': 10,
                                                    'test_size': None,
                                                    'random_state': None}
                                   }


def _get_y_array(y, task_type):
    if task_type in CLASSIFICATION_TASKS and task_type != \
            MULTICLASSMULTIOUTPUT:
        return y.ravel()
    else:
        return y


# # TODO: Add subsample to dataset directory
# def subsample_indices(train_indices, subsample, task_type, Y_train):
#
#     if not isinstance(subsample, float):
#         raise ValueError(
#             'Subsample must be of type float, but is of type %s'
#             % type(subsample)
#         )
#     elif subsample > 1:
#         raise ValueError(
#             'Subsample must not be larger than 1, but is %f'
#             % subsample
#         )
#
#     if subsample is not None and subsample < 1:
#         # Only subsample if there are more indices given to this method than
#         # required to subsample because otherwise scikit-learn will complain
#
#         if task_type in CLASSIFICATION_TASKS and task_type != MULTILABEL_CLASSIFICATION:
#             stratify = Y_train[train_indices]
#         else:
#             stratify = None
#
#         indices = np.arange(len(train_indices))
#         cv_indices_train, _ = train_test_split(
#             indices,
#             stratify=stratify,
#             train_size=subsample,
#             random_state=1,
#             shuffle=True,
#         )
#         train_indices = train_indices[cv_indices_train]
#         return train_indices
#
#     return train_indices


class TrainEvaluator(AbstractEvaluator):
    def __init__(self, backend, queue, metric,
                 configuration=None,
                 all_supported_metrics=False,
                 seed=1,
                 output_y_hat_optimization=True,
                 resampling_strategy=None,
                 resampling_strategy_args=None,
                 num_run=None,
                 budget=None,
                 budget_type=None,
                 keep_models=False,
                 include=None,
                 exclude=None,
                 disable_file_output=False,
                 init_params=None, ):
        super().__init__(
            backend=backend,
            queue=queue,
            configuration=configuration,
            metric=metric,
            all_supported_metrics=all_supported_metrics,
            seed=seed,
            output_y_hat_optimization=output_y_hat_optimization,
            num_run=num_run,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output,
            init_params=init_params,
            budget=budget,
            budget_type=budget_type,
        )

        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            self.resampling_strategy_args = {}
        else:
            self.resampling_strategy_args = resampling_strategy_args
        self.splitter = self.get_splitter(self.datamanager)
        self.num_cv_folds = self.splitter.get_n_splits(
            groups=self.resampling_strategy_args.get('groups')
        )

        self.Y_optimization = None
        self.Y_targets = [None] * self.num_cv_folds
        self.Y_train_targets = np.ones(self.y_train.shape) * np.NaN
        self.models = [None] * self.num_cv_folds
        self.indices = [None] * self.num_cv_folds

        self.keep_models = keep_models

    def fit_predict_and_loss(self):
        """Fit, predict and compute the loss for cross-validation and
        holdout"""

        if self.num_cv_folds == 1:

            for train_split, test_split in self.splitter.split(
                self.X_train, self.y_train,
                groups=self.resampling_strategy_args.get('groups')
            ):
                self.Y_optimization = self.y_train[test_split]
                self.Y_actual_train = self.y_train[train_split]
                self._fit_and_predict(0, train_indices=train_split,
                                      test_indices=test_split,
                                      add_model_to_self=True)

    def _fit_and_predict(self, fold, train_indices, test_indices,
                                 add_model_to_self):
        model = self._get_model()

        self.indices[fold] = ((train_indices, test_indices))

        # Do only output the files in the case of iterative holdout,
        # In case of iterative partial cv, no file output is needed
        # because ensembles cannot be built
        file_output = True if self.num_cv_folds == 1 else False
        X = {'train_indices': train_indices, 'val_indices': test_indices, **self._init_params}  # fit dictionary
        y = None
        model = model.fit(X, y)
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
            test_pred=Y_test_pred,
            additional_run_info=additional_run_info,
            file_output=file_output,
            status=status,
        )

    def _predict(self, model, test_indices, train_indices):
        train_pred = self.predict_function(self.X_train[train_indices],
                                           model, self.task_type,
                                           self.y_train[train_indices])

        valid_pred = self.predict_function(self.X_train[test_indices],
                                         model, self.task_type,
                                         self.y_train[train_indices])

        if self.X_test is not None:
            X_test = self.X_test.copy()
            test_pred = self.predict_function(X_test, model,
                                              self.task_type,
                                              self.y_train[train_indices])
        else:
            test_pred = None

        return train_pred, valid_pred, test_pred

    def get_splitter(self, D):

        if self.resampling_strategy_args is None:
            self.resampling_strategy_args = {}

        if not isinstance(self.resampling_strategy, str):

            if issubclass(self.resampling_strategy, BaseCrossValidator) or \
               issubclass(self.resampling_strategy, _RepeatedSplits) or \
               issubclass(self.resampling_strategy, BaseShuffleSplit):

                class_name = self.resampling_strategy.__name__
                if class_name not in __baseCrossValidator_defaults__:
                    raise ValueError('Unknown CrossValidator.')
                ref_arg_dict = __baseCrossValidator_defaults__[class_name]

                y = D.data['Y_train']
                if (D.info['task'] in CLASSIFICATION_TASKS and
                    STRING_TO_OUTPUT_TYPES[D.info['output_type']] != MULTICLASSMULTIOUTPUT) or \
                   (D.info['task'] in REGRESSION_TASKS and
                    STRING_TO_OUTPUT_TYPES[D.info['output_type']] != CONTINUOUSMULTIOUTPUT):

                    y = y.ravel()
                if class_name == 'PredefinedSplit':
                    if 'test_fold' not in self.resampling_strategy_args:
                        raise ValueError('Must provide parameter test_fold'
                                         ' for class PredefinedSplit.')
                if class_name == 'LeaveOneGroupOut' or \
                        class_name == 'LeavePGroupsOut' or\
                        class_name == 'GroupKFold' or\
                        class_name == 'GroupShuffleSplit':
                    if 'groups' not in self.resampling_strategy_args:
                        raise ValueError('Must provide parameter groups '
                                         'for chosen CrossValidator.')
                    try:
                        if self.resampling_strategy_args['groups'].shape[0] != y.shape[0]:
                            raise ValueError('Groups must be array-like '
                                             'with shape (n_samples,).')
                    except Exception:
                        raise ValueError('Groups must be array-like '
                                         'with shape (n_samples,).')
                else:
                    if 'groups' in self.resampling_strategy_args:
                        if self.resampling_strategy_args['groups'].shape[0] != y.shape[0]:
                            raise ValueError('Groups must be array-like'
                                             ' with shape (n_samples,).')

                # Put args in self.resampling_strategy_args
                for key in ref_arg_dict:
                    if key == 'n_splits':
                        if 'folds' not in self.resampling_strategy_args:
                            self.resampling_strategy_args['folds'] = ref_arg_dict['n_splits']
                    else:
                        if key not in self.resampling_strategy_args:
                            self.resampling_strategy_args[key] = ref_arg_dict[key]

                # Instantiate object with args
                init_dict = copy.deepcopy(self.resampling_strategy_args)
                init_dict.pop('groups', None)
                if 'folds' in init_dict:
                    init_dict['n_splits'] = init_dict.pop('folds', None)
                cv = copy.deepcopy(self.resampling_strategy)(**init_dict)

                if 'groups' not in self.resampling_strategy_args:
                    self.resampling_strategy_args['groups'] = None

                return cv

        y = D.data['Y_train']
        shuffle = self.resampling_strategy_args.get('shuffle', True)
        train_size = 0.67
        if self.resampling_strategy_args:
            train_size = self.resampling_strategy_args.get('train_size',
                                                           train_size)
        test_size = float("%.4f" % (1 - train_size))

        if D.info['task'] in CLASSIFICATION_TASKS and \
                STRING_TO_OUTPUT_TYPES[D.info['output_type']] != MULTICLASSMULTIOUTPUT:

            y = y.ravel()
            if self.resampling_strategy in ['holdout']:

                if shuffle:
                    try:
                        cv = StratifiedShuffleSplit(n_splits=1,
                                                    test_size=test_size,
                                                    random_state=1)
                        test_cv = copy.deepcopy(cv)
                        next(test_cv.split(y, y))
                    except ValueError as e:
                        if 'The least populated class in y has only' in e.args[0]:
                            cv = ShuffleSplit(n_splits=1, test_size=test_size,
                                              random_state=1)
                        else:
                            raise e
                else:
                    tmp_train_size = int(np.floor(train_size * y.shape[0]))
                    test_fold = np.zeros(y.shape[0])
                    test_fold[:tmp_train_size] = -1
                    cv = PredefinedSplit(test_fold=test_fold)
                    cv.n_splits = 1  # As sklearn is inconsistent here
            elif self.resampling_strategy in ['cv']:
                if shuffle:
                    cv = StratifiedKFold(
                        n_splits=self.resampling_strategy_args['folds'],
                        shuffle=shuffle, random_state=1)
                else:
                    cv = KFold(n_splits=self.resampling_strategy_args['folds'],
                               shuffle=shuffle)
            else:
                raise ValueError(self.resampling_strategy)
        else:
            if self.resampling_strategy in ['holdout']:
                # TODO shuffle not taken into account for this
                if shuffle:
                    cv = ShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=1)
                else:
                    tmp_train_size = int(np.floor(train_size * y.shape[0]))
                    test_fold = np.zeros(y.shape[0])
                    test_fold[:tmp_train_size] = -1
                    cv = PredefinedSplit(test_fold=test_fold)
                    cv.n_splits = 1  # As sklearn is inconsistent here
            elif self.resampling_strategy in ['cv', 'partial-cv',
                                              'partial-cv-iterative-fit']:
                random_state = 1 if shuffle else None
                cv = KFold(
                    n_splits=self.resampling_strategy_args['folds'],
                    shuffle=shuffle,
                    random_state=random_state,
                )
            else:
                raise ValueError(self.resampling_strategy)
        return cv


# create closure for evaluating an algorithm
def eval_holdout(
        queue,
        config,
        backend,
        resampling_strategy,
        resampling_strategy_args,
        metric,
        seed,
        num_run,
        instance,
        all_scoring_functions,
        output_y_hat_optimization,
        include,
        exclude,
        disable_file_output,
        init_params=None,
        budget=100.0,
        budget_type=None,
):
    evaluator = TrainEvaluator(
        backend=backend,
        queue=queue,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
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
    )
    evaluator.fit_predict_and_loss()

