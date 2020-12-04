# -*- encoding: utf-8 -*-
from typing import Any, Dict, List, Optional

from autoPyTorch.constants import (
    CLASSIFICATION_TASKS,
    IMAGE_TASKS,
    REGRESSION_TASKS,
    TABULAR_TASKS,
)
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.pipeline.tabular_regression import TabularRegressionPipeline
from autoPyTorch.utils.common import FitRequirement

__all__ = [
    'get_dataset_requirements',
]


def get_dataset_requirements(info: Dict[str, Any],
                             include_estimators: Optional[List[str]] = None,
                             exclude_estimators: Optional[List[str]] = None,
                             include_preprocessors: Optional[List[str]] = None,
                             exclude_preprocessors: Optional[List[str]] = None
                             ) -> List[FitRequirement]:
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

    task_type = info['task_type']
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
        return _get_regression_dataset_requirements(info, include, exclude)
    else:
        return _get_classification_dataset_requirements(info, include, exclude)


def _get_regression_dataset_requirements(info: Dict[str, Any], include: Dict[str, List[str]],
                                         exclude: Dict[str, List[str]]) -> List[FitRequirement]:
    task_type = info['task_type']
    output_type = info['output_type']
    sparse = info['is_sparse']
    dataset_properties = {
        'task_type': task_type,
        'output_type': output_type,
        'sparse': sparse,
    }
    if task_type in TABULAR_TASKS:
        dataset_properties.update({'numerical_columns': info['numerical_columns'],
                                   'categorical_columns': info['categorical_columns']})
        fit_requirements = TabularRegressionPipeline(
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude
        ).get_dataset_requirements()
        return fit_requirements
    else:
        raise ValueError("Task_type not supported")


def _get_classification_dataset_requirements(info: Dict[str, Any], include: Dict[str, List[str]],
                                             exclude: Dict[str, List[str]]) -> List[FitRequirement]:
    task_type = info['task_type']
    output_type = info['output_type']
    sparse = info['issparse']

    dataset_properties = {
        'task_type': task_type,
        'output_type': output_type,
        'sparse': sparse
    }
    if task_type in TABULAR_TASKS:
        dataset_properties.update({'numerical_columns': info['numerical_columns'],
                                   'categorical_columns': info['categorical_columns']})
        return TabularClassificationPipeline(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude).\
            get_dataset_requirements()
    elif task_type in IMAGE_TASKS:
        return ImageClassificationPipeline(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude).\
            get_dataset_requirements()
    else:
        raise ValueError("Task_type not supported")
