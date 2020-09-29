from typing import Any, Dict, Optional

from pytorch_lightning.metrics import classification
from pytorch_lightning.metrics.metric import Metric

import torch.tensor

from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


class PrecisionRecall(autoPyTorchMetric):
    """
    Computes the precision, recall scores for classification
    Args:
        pos_label (int):
        reduce_group:
        reduce_op:
    Returns:
        precision (torch.tensor)
        recall (torch.tensor)
        threshold (torch.tensor)
    """

    def __init__(self,
                 pos_label: int = 1,
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None
                 ):
        super().__init__()
        self.pos_label = pos_label
        self.reduce_op = reduce_op
        self.reduce_group = reduce_group
        self.metric: Metric = classification.PrecisionRecall(pos_label=self.pos_label,
                                                             reduce_op=self.reduce_op,
                                                             reduce_group=self.reduce_group)

    def __call__(self,
                 predictions: torch.tensor,
                 targets: torch.tensor
                 ) -> torch.tensor:
        return self.metric(predictions, targets)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'PR',
            'name': 'Precision Recall',
            'task_type': 'classification'
        }
