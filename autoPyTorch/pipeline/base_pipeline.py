from abc import ABCMeta
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state


class BasePipeline(Pipeline):
    """Base class for all pipeline objects.
    Notes
    -----
    This class should not be instantiated, only subclassed."""
    __metaclass__ = ABCMeta

    def __init__(
        self,
        config: Optional[Configuration] = None,
        steps: Optional[List[Tuple[str, AutoSklearnChoice]]] = None,
        include: Optional[Dict] = None,
        exclude: Optional[Dict] = None,
        random_state: Optional[np.random.RandomState] = None,
        init_params: Optional[Dict] = None
    ):

        self.init_params = init_params if init_params is not None else {}
        self.include = include if include is not None else {}
        self.exclude = exclude if exclude is not None else {}

        if steps is None:
            self.steps = self._get_pipeline_steps()
        else:
            self.steps = steps

        self.config_space = self.get_hyperparameter_search_space()

        if config is None:
            self.config = self.config_space.get_default_configuration()
        else:
            if isinstance(config, dict):
                config = Configuration(self.config_space, config)
            if self.config_space != config.configuration_space:
                print(self.config_space._children)
                print(config.configuration_space._children)
                import difflib
                diff = difflib.unified_diff(
                    str(self.config_space).splitlines(),
                    str(config.configuration_space).splitlines())
                diff = '\n'.join(diff)
                raise ValueError('Configuration passed does not come from the '
                                 'same configuration space. Differences are: '
                                 '%s' % diff)
            self.config = config

        self.set_hyperparameters(self.config, init_params=init_params)

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)
        super().__init__(steps=self.steps)

        self._additional_run_info = {}

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            **fit_params: Any
            ) -> Pipeline:
        """Fit the selected algorithm to the training data.

        Args:
            X (np.ndarray): Training data. The preferred type of the matrix (dense or sparse)
                depends on the estimator selected.
            y (np.ndarray): array-like Targets
            fit_params (dict): See the documentation of sklearn.pipeline.Pipeline for formatting
                instructions.

        Returns:
            self : returns an instance of self.
        """
        raise NotImplementedError()

    def fit_estimator(self, X, y, **fit_params):
        fit_params = {key.replace(":", "__"): value for key, value in
                      fit_params.items()}
        self._final_estimator.fit(X, y, **fit_params)
        return self

    def get_max_iter(self) -> int:
        if self.estimator_supports_iterative_fit():
            return self._final_estimator.get_max_iter()
        else:
            raise NotImplementedError()

    def configuration_fully_fitted(self) -> bool:
        return self._final_estimator.configuration_fully_fitted()

    def get_current_iter(self) -> int:
        return self._final_estimator.get_current_iter()

    def predict(self, X: np.ndarray, batch_size: Optional[int] = None
                ) -> np.ndarray:
        """Predict the classes using the selected model.

        Args:
            X (np.ndarray): input data to the array
            batch_size (Optional[int]): batch_size controls whether the pipeline will be
                called on small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.

        Returns:
            np.ndarray: the predicted values given input X
        """

        if batch_size is None:
            return super().predict(X).astype(self._output_dtype)
        else:
            if not isinstance(batch_size, int):
                raise ValueError("Argument 'batch_size' must be of type int, "
                                 "but is '%s'" % type(batch_size))
            if batch_size <= 0:
                raise ValueError("Argument 'batch_size' must be positive, "
                                 "but is %d" % batch_size)

            else:
                if self.num_targets == 1:
                    y = np.zeros((X.shape[0],), dtype=self._output_dtype)
                else:
                    y = np.zeros((X.shape[0], self.num_targets),
                                 dtype=self._output_dtype)

                # Copied and adapted from the scikit-learn GP code
                for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                                                  batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    y[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to], batch_size=None)

                return y

    def set_hyperparameters(
        self,
        configuration: Configuration,
        init_params: Optional[Dict] = None
    ) -> 'Pipeline':
        """Method to overwrite the default hyperparamter configuration of the pipeline

        Args:
            configuration (Configuration): configuration object to search and overwrite in
                the pertinent spaces
            init_params (Optional[Dict]): optional initial settings for the config

        """
        self.configuration = configuration

        for node_idx, n_ in enumerate(self.steps):
            node_name, node = n_

            sub_configuration_space = node.get_hyperparameter_search_space()
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('%s:' % node_name):
                    value = configuration[param]
                    new_name = param.replace('%s:' % node_name, '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space,
                                              values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('%s:' % node_name):
                        value = init_params[param]
                        new_name = param.replace('%s:' % node_name, '', 1)
                        sub_init_params_dict[new_name] = value
            else:
                sub_init_params_dict = None

            if isinstance(node, (AutoPytorchComponent, BasePipeline)):
                node.set_hyperparameters(configuration=sub_configuration,
                                         init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(self) -> ConfigurationSpace:
        """Return the configuration space for the CASH problem.

        Returns:
            Configuration: The configuration space describing the AutoSklearnClassifier.
        """
        if not hasattr(self, 'config_space') or self.config_space is None:
            self.config_space = self._get_hyperparameter_search_space(
                include=self.include, exclude=self.exclude,
                )
        return self.config_space

    def _get_hyperparameter_search_space(self,
                                         include: Optional[Dict] = None,
                                         exclude: Optional[Dict] = None,
                                         ) -> ConfigurationSpace:
        """Return the configuration space for the CASH problem.
        This method should be called by the method
        get_hyperparameter_search_space of a subclass. After the subclass
        assembles a list of available estimators and preprocessor components,
        _get_hyperparameter_search_space can be called to do the work of
        creating the actual
        ConfigSpace.configuration_space.ConfigurationSpace object.
        Args:
            include (Dict): Overwrite to include user desired components to the pipeline
            exclude (Dict): Overwrite to exclude user desired components to the pipeline

        Returns:
            Configuration: The configuration space describing the AutoPytorch estimator.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Retrieves a str representation of the current pipeline

        Returns:
            str: A formatted representation of the pipeline stages
                 and components
        """
        raise NotImplementedError()

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoPytorchComponent]]:
        raise NotImplementedError()

    def _get_estimator_hyperparameter_name(self) -> str:
        raise NotImplementedError()

    def get_additional_run_info(self) -> Dict:
        """Allows retrieving additional run information from the pipeline.
        Can be overridden by subclasses to return additional information to
        the optimization algorithm.

        Returns:
            Dict: Additional information about the pipeline
        """
        return self._additional_run_info
