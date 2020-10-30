from typing import Any, Dict, Optional

from pytorch_lightning.metrics import classification
from pytorch_lightning.metrics.metric import Metric

from sklearn.utils.multiclass import type_of_target

import torch.tensor

from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


class Accuracy(autoPyTorchMetric):
    def __init__(self,
                 reduction: str = 'elementwise_mean',
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None):
        super().__init__()
        self.class_reduction = reduction
        self.reduce_group = reduce_group
        self.reduce_op = reduce_op
        # Accuracy(threshold=0.5, compute_on_step=True, dist_sync_on_step=False, process_group=None)
        self.metric: Metric = classification.Accuracy()

    def __call__(self,
                 predictions: torch.tensor,
                 targets: torch.tensor
                 ) -> torch.tensor:

        type_targets = type_of_target(targets)

        # If dealing with binary predictions, the crossentropy softmax
        # creates a prediction with 2 dimensionality (i.e. the probability of
        # the first and second class). The targets are one dimensional, so
        # we translated the predictions from a 1-encode context to a class
        if type_targets in ['binary', 'multiclass']:
            predictions = torch.argmax(predictions, dim=1)

        return self.metric(predictions, targets)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'Acc',
            'name': 'Accuracy',
            'task_type': 'classification',
            'objective': 'maximise'
        }
