# -*- encoding: utf-8 -*-
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from sklearn.pipeline import Pipeline

from autoPyTorch.constants import (
    IMAGE_TASKS,
    TABULAR_TASKS,
    BINARY,
    CLASSIFICATION_TASKS,
    MULTICLASS,
    MULTICLASSMULTIOUTPUT,
    CONTINUOUSMULTIOUTPUT,
    CONTINUOUS,
    REGRESSION_TASKS,
    STRING_TO_TASK_TYPES
)
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline


__all__ = [
    'get_configuration_space',
    # 'get_class',
]


def get_configuration_space(info: Dict[str, Any],
                            include_estimators: Optional[List[str]] = None,
                            exclude_estimators: Optional[List[str]] = None,
                            include_preprocessors: Optional[List[str]] = None,
                            exclude_preprocessors: Optional[List[str]] = None
                            ) -> ConfigurationSpace:
    exclude = dict()
    include = dict()
    if include_preprocessors is not None and \
            exclude_preprocessors is not None:
        raise ValueError('Cannot specify include_preprocessors and '
                         'exclude_preprocessors.')
    elif include_preprocessors is not None:
        include['feature_preprocessor'] = include_preprocessors
    elif exclude_preprocessors is not None:
        exclude['feature_preprocessor'] = exclude_preprocessors

    task_type = STRING_TO_TASK_TYPES[info['task_type']]
    if include_estimators is not None and \
            exclude_estimators is not None:
        raise ValueError('Cannot specify include_estimators and '
                         'exclude_estimators.')
    elif include_estimators is not None:
        if task_type in CLASSIFICATION_TASKS:
            include['classifier'] = include_estimators
        elif task_type in REGRESSION_TASKS:
            include['regressor'] = include_estimators
        else:
            raise ValueError(info['task_type'])
    elif exclude_estimators is not None:
        if task_type in CLASSIFICATION_TASKS:
            exclude['classifier'] = exclude_estimators
        elif info['task_type'] in REGRESSION_TASKS:
            exclude['regressor'] = exclude_estimators
        else:
            raise ValueError(info['task_type'])

    if task_type in REGRESSION_TASKS:
        return _get_regression_configuration_space(info, include, exclude)
    else:
        return _get_classification_configuration_space(info, include, exclude)


def _get_regression_configuration_space(info: Dict[str, Any], include: Dict[str, List[str]],
                                        exclude: Dict[str, List[str]]) -> ConfigurationSpace:
    task_type = STRING_TO_TASK_TYPES[info['task_type']]
    output_type = info['output_type']
    sparse = False
    if info['is_sparse'] == 1:
        sparse = True
    dataset_properties = {
        'task_type': task_type,
        'output_type': output_type,
        'sparse': sparse,
    }
    if task_type in TABULAR_TASKS:

        dataset_properties.update({'categories': info['categories'],
                                   'numerical_columns':  info['numerical_columns'],
                                   'categorical_columns': info['categorical_columns']})
        configuration_space = TabularRegressionPipeline(
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude
        ).get_hyperparameter_search_space()
        return configuration_space
    else:
        raise ValueError("Task_type not supported")


def _get_classification_configuration_space(info: Dict[str, Any], include: Dict[str, List[str]],
                                            exclude: Dict[str, List[str]]) -> ConfigurationSpace:
    task_type = STRING_TO_TASK_TYPES[info['task_type']]
    output_type = info['output_type']
    sparse = False
    if info['is_sparse'] == 1:
        sparse = True

    dataset_properties = {
        'task_type': task_type,
        'output_type': output_type,
        'sparse': sparse
    }
    if task_type in TABULAR_TASKS:
        dataset_properties.update({'categories': info['categories'],
                                   'numerical_columns': info['numerical_columns'],
                                   'categorical_columns': info['categorical_columns']})
        return TabularClassificationPipeline(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude).\
            get_hyperparameter_search_space()
    elif task_type in IMAGE_TASKS:
        return ImageClassificationPipeline(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude).\
            get_hyperparameter_search_space()
    else:
        raise ValueError("Task_type not supported")


# def get_class(info: Dict[str, Any]) -> Pipeline:
#     if info['task_type'] in REGRESSION_TASKS:
#         return SimpleRegressionPipeline
#     else:
#         return SimpleClassificationPipeline
