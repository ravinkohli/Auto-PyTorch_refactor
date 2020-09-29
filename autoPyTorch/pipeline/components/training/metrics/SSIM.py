from typing import Any, Dict, Optional, Tuple

from pytorch_lightning.metrics import regression
from pytorch_lightning.metrics.metric import Metric

import torch.tensor

from autoPyTorch.pipeline.components.training.metrics.base_metric import autoPyTorchMetric


class SSIM(autoPyTorchMetric):
    def __init__(self,
                 kernel_size: Tuple[int, int] = (11, 11),
                 sigma: Tuple[float, float] = (1.5, 1.5),
                 reduction: str = 'elementwise_mean',
                 data_range: Optional[float] = None,
                 k1: float = 0.01,
                 k2: float = 0.03
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.metric: Metric = regression.SSIM(kernel_size=self.kernel_size,
                                              sigma=self.sigma,
                                              reduction=self.reduction,
                                              data_range=self.data_range,
                                              k1=self.k1,
                                              k2=self.k2)

    def __call__(self,
                 predictions: torch.tensor,
                 targets: torch.tensor
                 ) -> torch.tensor:
        return self.metric(predictions, targets)

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {
            'shortname': 'Metric',
            'name': 'Structural Similarity Index Measure',
            'task_type': 'regression'
        }
