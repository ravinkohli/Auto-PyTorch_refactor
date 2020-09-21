from typing import Any, Dict, Optional

import torch.tensor

from pytorch_lightning.metrics import classification
from pytorch_lightning.metrics.metric import Metric

from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


class ConfusionMatrix(autoPyTorchMetric):
    def __init__(self):
        self.metric: Metric = classification.ConfusionMatrix()

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
