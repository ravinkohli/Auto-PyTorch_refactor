from typing import Optional

from pytorch_lightning.metrics.metric import Metric

from sklearn.base import BaseEstimator

import torch.tensor


class autoPyTorchMetric(BaseEstimator):
    def __init__(self):
        self.metric: Optional[Metric] = None

    def __call__(self,
                 predictions: torch.tensor,
                 targets: torch.tensor
                 ) -> torch.tensor:
        raise NotImplementedError()

    def get_metric(self) -> Metric:
        return self.metric
