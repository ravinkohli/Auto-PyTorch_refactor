from typing import Any, Dict, Iterable, Optional

from pytorch_lightning.metrics import classification
from pytorch_lightning.metrics import regression
from pytorch_lightning.metrics.metric import Metric

autopytorch_metrics = dict({
    'classification': {
        'Accuracy': classification.Accuracy,
        'AveragePrecision': classification.AveragePrecision,
        'AUROC': classification.AUROC,
        'ConfusionMatrix': classification.ConfusionMatrix,
        'F1': classification.F1,
        'MulticlassPrecisionRecall': classification.MulticlassPrecisionRecall,
        'MulticlassROC': classification.MulticlassROC,
        'Precision': classification.Precision,
        'PrecisionRecall': classification.PrecisionRecall,
        'Recall': classification.Recall,
        'ROC': classification.ROC
    },
    'regression': {
        'RMSE': regression.RMSE,
        'RMSLE': regression.RMSLE,
        'SSIM': regression.SSIM,
    },
})


def get_supported_metrics(dataset_properties: Dict[str, Any]) -> Dict[str, Metric]:
    supported_metrics = dict()

    for key, value in autopytorch_metrics[dataset_properties['task_type'].split('_')[-1]].items():
        supported_metrics[key] = value

    return supported_metrics


def get_metric_instances(dataset_properties: Dict[str, Any], names: Optional[Iterable[str]] = None) -> Iterable[Metric]:
    assert 'task_type' in dataset_properties, \
        "Expected dataset_properties to have task_type got {}".format(dataset_properties.keys())

    task_type = dataset_properties['task_type']
    supported_metrics = get_supported_metrics(dataset_properties)
    metrics = list()
    if names is not None:
        for name in names:
            if name not in supported_metrics.keys():
                raise ValueError("Invalid name entered for task {}, and output type {} "
                                 "currently supported metrics for task include {}".format(
                    task_type, dataset_properties['output_type'], list(supported_metrics.keys())))
            else:
                metric = supported_metrics[name]
                metrics.append(metric)
    else:
        metrics = [supported_metrics[key] for key in supported_metrics.keys()]

    return metrics
