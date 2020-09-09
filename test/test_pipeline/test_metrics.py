import unittest

from pytorch_lightning.metrics.metric import Metric

import torch

from autoPyTorch.pipeline.components.training.metrics import get_metric_instances
from autoPyTorch.pipeline.components.training.metrics import get_supported_metrics

class MetricsTest(unittest.TestCase):
    def test_get_no_name(self):
        dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multi-class'}
        metrics = get_metric_instances(dataset_properties)
        for metric in metrics:
            self.assertTrue(issubclass(metric, Metric))

    def test_get_name(self):
        dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multi-class'}
        names = ['Accuracy', 'AveragePrecision']
        metrics = get_metric_instances(dataset_properties, names)
        for i in range(len(metrics)):
            self.assertTrue(issubclass(metrics[i], Metric))
            self.assertEqual(metrics[i].__name__.lower(), names[i].lower())

    def test_get_name_error(self):
        dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multi-class'}
        names = ['RMSE', 'AveragePrecision']
        try:
            get_metric_instances(dataset_properties, names)
        except ValueError as msg:
            self.assertRegex(str(msg), r"Invalid name entered for task [a-z]+_[a-z]+, "
                                       r"and output type [a-z]+-[a-z]+ currently supported metrics for task include .*")

    def test_metrics(self):
        dataset_properties = {'task_type': 'tabular_classification', 'output_type': 'multi-class'}
        y_target = torch.tensor([0, 1, 3, 2])
        y_pred = torch.empty(4, dtype=torch.int).random_(4)
        supported_metrics = get_supported_metrics(dataset_properties=dataset_properties)
        skip_metrics = ['MulticlassPrecisionRecall', 'MulticlassROC', 'PrecisionRecall', 'ROC']
        for key, value in supported_metrics.items():
            if key in skip_metrics:
                continue
            metric = value()
            score = metric(y_pred, y_target)
            self.assertIsInstance(score, torch.Tensor)
