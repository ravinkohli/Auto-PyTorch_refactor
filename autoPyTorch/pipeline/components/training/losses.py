from typing import Any, Dict, Optional

from torch.nn.modules.loss import (
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    MSELoss,
    L1Loss
)
from torch.nn.modules.loss import _Loss as Loss


def get_default(x: Dict[str, Loss]) -> Loss:
    return x[list(x.keys())[0]]


losses = dict(classification=dict(
    CrossEntropyLoss=dict(
        module=CrossEntropyLoss, supported_output_type='multi-class'),
    BCEWithLogitsLoss=dict(
        module=BCEWithLogitsLoss, supported_output_type='binary-class')),
    regression=dict(
        MSELoss=dict(
            module=MSELoss, supported_output_type='continuous'),
        L1Loss=dict(
            module=L1Loss, supported_output_type='continuous')))


def get_supported_losses(dataset_properties: Dict[str, Any]) -> Dict[str, Loss]:
    supported_losses = dict()
    for key, value in losses[dataset_properties['task_type'].split('_')[-1]].items():
        if value['supported_output_type'] == dataset_properties['output_type']:
            supported_losses[key] = value['module']

    return supported_losses


def get_loss_instance(dataset_properties: Dict[str, Any], name: Optional[str] = None) -> Loss:
    assert 'task_type' in dataset_properties, \
        "Expected dataset_properties to have task_type got {}".format(dataset_properties.keys())
    assert 'output_type' in dataset_properties, \
        "Expected dataset_properties to have output_type got {}".format(dataset_properties.keys())

    task_type = dataset_properties['task_type']
    supported_losses = get_supported_losses(dataset_properties)

    if name is not None:
        if name not in supported_losses.keys():
            raise ValueError("Invalid name entered for task {}, and output type {} currently supported losses"
                             " for task include {}".format(task_type,
                                                           dataset_properties['output_type'],
                                                           list(supported_losses.keys())))
        else:
            loss = supported_losses[name]
    else:
        loss = get_default(supported_losses)

    return loss()
