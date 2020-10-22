import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional

from autoPyTorch.constants import CLASSIFICATION_TASKS, REGRESSION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    find_components,
)
from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


metrics_directory = os.path.split(__file__)[0]
_metrics = find_components(__package__,
                            metrics_directory,
                            autoPyTorchMetric)
_addons = ThirdPartyComponents(autoPyTorchMetric)


def add_metric(metric: autoPyTorchMetric) -> None:
    _addons.add_component(metric)


def get_components() -> Dict[str, autoPyTorchMetric]:
    """Returns the available metric components

    Args:
        None

    Returns:
        Dict[str, autoPyTorchMetric]: all autoPyTorchMetric components available
            as choices
    """
    components = OrderedDict()
    components.update(_metrics)
    components.update(_addons.components)
    return components


def get_supported_metrics(dataset_properties: Dict[str, Any]) -> Dict[str, autoPyTorchMetric]:
    supported_metrics = dict()

    task_type = dataset_properties['task_type']
    components = get_components()

    for name, component in components.items():
        if component.get_properties(dataset_properties)['task_type'] in task_type:
            supported_metrics.update({name: component})

    return supported_metrics


def get_metrics(dataset_properties: Dict[str, Any],
                names: Optional[Iterable[str]] = None
                ) -> List[autoPyTorchMetric]:

    assert 'task_type' in dataset_properties, \
        "Expected dataset_properties to have task_type got {}".format(dataset_properties.keys())

    default_metrics = dict(classification='Accuracy',
                           regression='RMSE')

    supported_metrics = get_supported_metrics(dataset_properties)
    metrics = list()  # type: List[autoPyTorchMetric]
    if names is not None:
        for name in names:
            if name not in supported_metrics.keys():
                raise ValueError("Invalid name entered for task {}, currently "
                                 "supported metrics for task include {}".format(dataset_properties['task_type'],
                                                                                list(supported_metrics.keys())))
            else:
                metric = supported_metrics[name]
                metrics.append(metric)
    else:
        if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
            metrics.append(supported_metrics[default_metrics['classification']])
        if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in REGRESSION_TASKS:
            metrics.append(supported_metrics[default_metrics['regression']])

    return metrics
