import unittest
import unittest.mock

import torch

import autoPyTorch.pipeline.components.training.metrics.utils as metric_components
from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


class MetricsTest(unittest.TestCase):
    def test_get_no_name(self):
        dataset_properties = {'task_type': 'tabular_classification'}
        metrics = metric_components.get_metrics(dataset_properties)
        for metric in metrics:
            self.assertTrue(issubclass(metric, autoPyTorchMetric))

    def test_get_name(self):
        dataset_properties = {'task_type': 'tabular_classification'}
        names = ['Accuracy', 'AveragePrecision']
        metrics = metric_components.get_metrics(dataset_properties, names)
        for i in range(len(metrics)):
            self.assertTrue(issubclass(metrics[i], autoPyTorchMetric))
            self.assertEqual(metrics[i].__name__.lower(), names[i].lower())

    def test_get_name_error(self):
        dataset_properties = {'task_type': 'tabular_classification'}
        names = ['RMSE', 'AveragePrecision']
        try:
            metric_components.get_metrics(dataset_properties, names)
        except ValueError as msg:
            self.assertRegex(str(msg), r"Invalid name entered for task [a-z]+_[a-z]+, "
                                       r"currently supported metrics for task include .*")

    def test_get_properties(self):
        metric_dict = metric_components.get_components()
        for key, value in metric_dict.items():
            properties = value.get_properties()
            self.assertIn('name', properties.keys())
            self.assertIn('shortname', properties.keys())
            self.assertIn('task_type', properties.keys())
            self.assertIn('objective', properties.keys())

    def test_metrics(self):
        dataset_properties = {'task_type': 'tabular_classification'}
        y_target = torch.tensor([0, 1, 3, 2])
        y_pred = torch.empty(4, dtype=torch.int).random_(4)
        supported_metrics = metric_components.get_supported_metrics(dataset_properties=dataset_properties)
        for key, value in supported_metrics.items():
            metric = value()
            score = metric(y_pred, y_target)
            if key == 'PrecisionRecall':
                self.assertTrue(len(score) == 3)
                for i in range(3):
                    self.assertIsInstance(score[i], torch.Tensor)
            else:
                self.assertIsInstance(score, torch.Tensor)

    def test_add_metric(self):
        class DummyMetric(autoPyTorchMetric):
            def __init__(self):
                super().__init__()
                self.metric = unittest.mock.Mock

            def __call__(self,
                         predictions,
                         targets
                         ) -> torch.tensor:
                return torch.tensor(1)

            @staticmethod
            def get_properties(dataset_properties=None):
                return {
                    'shortname': 'Dummy',
                    'name': 'DummyMetric',
                    'task_type': 'classification',
                    'objective': 'maximise'
                }
        # No third party components to start with
        self.assertEqual(len(metric_components._addons.components), 0)

        # Then make sure the metric can be added and query'ed
        metric_components.add_metric(DummyMetric)
        self.assertEqual(len(metric_components._addons.components), 1)
        metrics = metric_components.get_components()
        self.assertIn('DummyMetric', metrics.keys())
