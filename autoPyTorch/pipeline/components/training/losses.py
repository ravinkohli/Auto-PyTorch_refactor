from typing import Any, Dict, Optional

from torch import nn

get_default = lambda x: x[list(x.keys())[0]]

losses = dict({'classification': {
    'CrossEntropyLoss': {'module': nn.CrossEntropyLoss,
                         'supported_output_type': 'multi-class'},
    'BCEWithLogitsLoss': {'module': nn.BCEWithLogitsLoss,
                          'supported_output_type': 'binary-class'}
},
    'regression': {
        'MSELoss': {'module': nn.MSELoss,
                    'supported_output_type': 'continuous'},
        'L1Loss': {'module': nn.L1Loss,
                   'supported_output_type': 'continuous'},
    }
})


def get_supported_losses(dataset_properties: Dict[str, Any]) -> Dict[str, nn.Module]:

    supported_losses = dict()
    for key, value in losses[dataset_properties['task_type'].split('_')[-1]].items():
        if value['supported_output_type'] == dataset_properties['output_type']:
            supported_losses[key] = value['module']

    return supported_losses


def get_loss_instance(dataset_properties: Dict[str, Any], name: Optional[str] = None) -> nn.Module:
    assert 'task_type' in dataset_properties, \
        "Expected dataset_properties to have task_type got {}".format(dataset_properties.keys())
    assert 'output_type' in dataset_properties, \
        "Expected dataset_properties to have output_type got {}".format(dataset_properties.keys())

    task_type = dataset_properties['task_type']
    supported_losses = get_supported_losses(dataset_properties)

    if name is not None:
        if name not in supported_losses.keys():
            raise ValueError("Invalid name entered for task {}, and output type {} "
                             "currently supported losses for task include {}".format(
                task_type, dataset_properties['output_type'], supported_losses.keys()))
        else:
            loss = supported_losses[name]
    else:
        loss = get_default(supported_losses)

    return loss()