from typing import Any, Dict, Optional

from pytorch_lightning.metrics import classification
from pytorch_lightning.metrics.metric import Metric

import torch.tensor

from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


class F1(autoPyTorchMetric):
    """
    Computes the F1 score, which is the harmonic mean of the precision and recall.
    It ranges between 1 and 0, where 1 is perfect and the worst value is 0.
    Args:
        num_classes (Optional[int]) – number of classes
        reduction (str) – a method for reducing accuracies over labels (default: takes the mean)
        reduce_group (Optional[Any]) – the process group to reduce metric results from DDP
        reduce_op (Optional[Any]) – the operation to perform for ddp reduction
    """
    def __init__(self,
                 num_classes: Optional[int] = None,
                 reduction: str = 'elementwise_mean',
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None):
        super().__init__()
        self.num_classes = num_classes
        self.class_reduction = reduction
        self.reduce_group = reduce_group
        self.reduce_op = reduce_op
        self.metric: Metric = classification.F1(num_classes=self.num_classes, reduction=self.class_reduction,
                                                reduce_group=self.reduce_group, reduce_op=self.reduce_op)

    def __call__(self,
                 predictions: torch.tensor,
                 targets: torch.tensor
                 ) -> torch.tensor:
        return self.metric(predictions, targets)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'F1',
            'name': 'F1 measure',
            'task_type': 'classification',
            'objective': 'maximise'
        }
