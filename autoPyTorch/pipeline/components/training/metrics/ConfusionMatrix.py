from typing import Any, Dict, Optional

from pytorch_lightning.metrics import classification
from pytorch_lightning.metrics.metric import Metric

import torch.tensor

from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


class ConfusionMatrix(autoPyTorchMetric):
    """
    Computes the confusion matrix
    Args:
        normalize (bool): True, to return a normalised confusion matrix
        reduce_group:
        reduce_op:
    Returns:
        (Tensor): A Tensor with the confusion matrix
    """

    def __init__(self,
                 normalize: bool = False,
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None
                 ):
        super().__init__()
        self.normalize = normalize
        self.reduce_op = reduce_op
        self.reduce_group = reduce_group
        self.metric: Metric = classification.ConfusionMatrix(normalize=self.normalize,
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
            'shortname': 'ConfusionMatrix',
            'name': 'Confusion Matrix',
            'task_type': 'classification',
            'objective': 'none'
        }
