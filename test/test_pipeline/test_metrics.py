import unittest

import torch

from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics
from autoPyTorch.pipeline.components.training.metrics.utils import get_supported_metrics


class MetricsTest(unittest.TestCase):
    def test_get_no_name(self):
        dataset_properties = {'task_type': 'tabular_classification'}
        metrics = get_metrics(dataset_properties)
        for metric in metrics:
            self.assertTrue(issubclass(metric, autoPyTorchMetric))

    def test_get_name(self):
        dataset_properties = {'task_type': 'tabular_classification'}
        names = ['Accuracy', 'AveragePrecision']
        metrics = get_metrics(dataset_properties, names)
        for i in range(len(metrics)):
            self.assertTrue(issubclass(metrics[i], autoPyTorchMetric))
            self.assertEqual(metrics[i].__name__.lower(), names[i].lower())

    def test_get_name_error(self):
        dataset_properties = {'task_type': 'tabular_classification'}
        names = ['RMSE', 'AveragePrecision']
        try:
            get_metrics(dataset_properties, names)
        except ValueError as msg:
            self.assertRegex(str(msg), r"Invalid name entered for task [a-z]+_[a-z]+, "
                                       r"currently supported metrics for task include .*")

    def test_metrics(self):
        dataset_properties = {'task_type': 'tabular_classification'}
        y_target = torch.tensor([0, 1, 3, 2])
        y_pred = torch.empty(4, dtype=torch.int).random_(4)
        supported_metrics = get_supported_metrics(dataset_properties=dataset_properties)
        for key, value in supported_metrics.items():
            metric = value()
            score = metric(y_pred, y_target)
            if key == 'PrecisionRecall':
                self.assertTrue(len(score) == 3)
                for i in range(3):
                    self.assertIsInstance(score[i], torch.Tensor)
            else:
                self.assertIsInstance(score, torch.Tensor)
