# -*- encoding: utf-8 -*-
from typing import List, Optional

import numpy as np

from scipy import sparse
from sklearn.utils.multiclass import type_of_target

from autoPyTorch.constants import STRING_TO_TASK_TYPES
from autoPyTorch.data.abstract_data_manager import AbstractDataManager


class XYDataManager(AbstractDataManager):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        task: int,
        feat_type: List[str],
        dataset_name: str
    ):
        super(XYDataManager, self).__init__(dataset_name)

        self.info['task_type'] = task
        if sparse.issparse(X):
            self.info['is_sparse'] = 1
            self.info['has_missing'] = np.all(np.isfinite(X.data))
        else:
            self.info['is_sparse'] = 0
            self.info['has_missing'] = np.all(np.isfinite(X))

        self.data['X_train'] = X
        self.data['Y_train'] = y
        if X_test is not None:
            self.data['X_test'] = X_test
        if y_test is not None:
            self.data['Y_test'] = y_test

        if feat_type is not None:
            for feat in feat_type:
                allowed_types = ['numerical', 'categorical']
                if feat.lower() not in allowed_types:
                    raise ValueError("Entry '%s' in feat_type not in %s" %
                                     (feat.lower(), str(allowed_types)))

        self.feat_type = feat_type


        if len(y.shape) > 2:
            raise ValueError('y must not have more than two dimensions, '
                             'but has %d.' % len(y.shape))

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of '
                             'datapoints, but have %d and %d.' % (X.shape[0],
                                                                  y.shape[0]))
        if self.feat_type is None:
            self.feat_type = ['Numerical'] * X.shape[1]
        if X.shape[1] != len(self.feat_type):
            raise ValueError('X and feat_type must have the same number of columns, '
                             'but are %d and %d.' %
                             (X.shape[1], len(self.feat_type)))
        categorical_columns = []
        numerical_columns = []
        for i, feat in enumerate(self.feat_type):
            if feat.lower() == 'numerical':
                numerical_columns.append(i)
            elif feat.lower() == 'categorical':
                categorical_columns.append(i)
            else:
                raise ValueError(feat)
        self.info['categorical_columns'] = categorical_columns
        self.info['numerical_columns'] = numerical_columns
        self.info['categories'] = [np.unique(self.data['X_train'][column]).tolist() for column in categorical_columns]
        self.info['output_type'] = type_of_target(y_test)
        self.info['is_small_preprocess'] = True
