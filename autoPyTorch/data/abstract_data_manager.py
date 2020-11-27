import abc
from typing import Any, Dict, List

import numpy as np


class AbstractDataManager():
    __metaclass__ = abc.ABCMeta

    def __init__(self, name: str):

        self._data = dict()  # type: Dict
        self._info = dict()  # type: Dict
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> Dict[str, np.ndarray]:
        return self._data

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    @property
    def feat_type(self) -> List[str]:
        return self._feat_type

    @feat_type.setter
    def feat_type(self, value: List[str]) -> None:
        self._feat_type = value
