from typing import Any, Dict, List

from sklearn.base import BaseEstimator


def get_tabular_preprocessers(X: Dict[str, Any]) -> Dict[str, List[BaseEstimator]]:
    preprocessor = dict(numerical=list(), categorical=list())  # type: Dict[str, List[BaseEstimator]]

    for key, value in X.items():
        if isinstance(value, dict):
            if 'numerical' or 'categorical' in value.keys():
                if isinstance(value['numerical'], BaseEstimator):
                    preprocessor['numerical'].append(value['numerical'])
                if isinstance(value['numerical'], BaseEstimator):
                    preprocessor['categorical'].append(value['categorical'])

    return preprocessor
