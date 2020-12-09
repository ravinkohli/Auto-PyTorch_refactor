import unittest
import unittest.mock

import numpy as np

from autoPyTorch.constants import (
    BINARY,
    CONTINUOUS,
    OUTPUT_TYPES_TO_STRING,
    STRING_TO_TASK_TYPES,
    TABULAR_CLASSIFICATION,
    TABULAR_REGRESSION,
    TASK_TYPES_TO_STRING
)
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_score, get_metrics


class MetricsTest(unittest.TestCase):
    def test_get_no_name(self):
        dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
                              'output_type': OUTPUT_TYPES_TO_STRING[BINARY]}
        metrics = get_metrics(dataset_properties)
        for metric in metrics:
            self.assertTrue(isinstance(metric, autoPyTorchMetric))

    def test_get_name(self):
        dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
                              'output_type': OUTPUT_TYPES_TO_STRING[BINARY]}
        names = ['accuracy', 'average_precision']
        metrics = get_metrics(dataset_properties, names)
        for i in range(len(metrics)):
            self.assertTrue(isinstance(metrics[i], autoPyTorchMetric))
            self.assertEqual(metrics[i].name.lower(), names[i].lower())

    def test_get_name_error(self):
        dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
                              'output_type': OUTPUT_TYPES_TO_STRING[BINARY]}
        names = ['root_mean_sqaured_error', 'average_precision']
        try:
            get_metrics(dataset_properties, names)
        except ValueError as msg:
            self.assertRegex(str(msg), r"Invalid name entered for task [a-z]+_[a-z]+, "
                                       r"currently supported metrics for task include .*")

    def test_metrics(self):
        # test of all classification metrics
        dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
                              'output_type': OUTPUT_TYPES_TO_STRING[BINARY]}
        y_target = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        metrics = get_metrics(dataset_properties=dataset_properties, all_supported_metrics=True)
        score_dict = calculate_score(y_pred, y_target, STRING_TO_TASK_TYPES[dataset_properties['task_type']], metrics)
        self.assertIsInstance(score_dict, dict)
        for name, score in score_dict.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(score, float)

        # test of all regression metrics
        dataset_properties = {'task_type': TASK_TYPES_TO_STRING[TABULAR_REGRESSION],
                              'output_type': OUTPUT_TYPES_TO_STRING[CONTINUOUS]}
        y_target = np.array([0.1, 0.6, 0.7, 0.4])
        y_pred = np.array([0.6, 0.7, 0.4, 1])
        metrics = get_metrics(dataset_properties=dataset_properties, all_supported_metrics=True)
        score_dict = calculate_score(y_pred, y_target, STRING_TO_TASK_TYPES[dataset_properties['task_type']], metrics)

        self.assertIsInstance(score_dict, dict)
        for name, score in score_dict.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(score, float)
