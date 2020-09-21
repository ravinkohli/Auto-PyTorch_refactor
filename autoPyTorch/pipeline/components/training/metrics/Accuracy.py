from typing import Any, Dict, Optional

import torch.tensor

from pytorch_lightning.metrics import classification
from pytorch_lightning.metrics.metric import Metric

from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


class Accuracy(autoPyTorchMetric):
    def __init__(self,
                 num_classes: Optional[int] = None,
                 reduction: str = 'micro',
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None):
        self.num_classes = num_classes
        self.class_reduction = reduction
        self.reduce_group = reduce_group
        self.reduce_op = reduce_op
        self.metric: Metric = classification.Accuracy(num_classes=self.num_classes, reduction=self.class_reduction,
                                                      reduce_group=self.reduce_group, reduce_op=self.reduce_op)

    def __call__(self,
                 predictions: torch.tensor,
                 targets: torch.tensor
                 ) -> torch.tensor:
        return self.metric(predictions, targets)

    @staticmethod
    def get_properties(cls, dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'Metric',
            'name': 'autopytorch metric {}'.format(cls.metric.__class__.__name__),
            'task_type': 'classification'
        }
