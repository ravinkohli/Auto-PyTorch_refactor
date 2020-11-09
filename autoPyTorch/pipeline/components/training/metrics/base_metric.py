from typing import Any, Dict, List, Optional

from pytorch_lightning.metrics.metric import Metric

import torch.tensor


class autoPyTorchMetric(object):
    _required_properties: List[str] = ['objective', 'task_type']

    def __init__(self) -> None:
        self.metric: Optional[Metric] = None

    @classmethod
    def get_required_properties(cls) -> List[str]:
        return cls._required_properties

    def __call__(self,
                 predictions: torch.tensor,
                 targets: torch.tensor
                 ) -> torch.tensor:
        raise NotImplementedError()

    def get_metric(self) -> Metric:
        return self.metric

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        raise NotImplementedError()
