import typing
from typing import Optional

from ConfigSpace.configuration_space import \
    Configuration, \
    ConfigurationSpace
import numpy as np


class Task:

    def __init__(
        self,
        **pipeline_kwargs,
    ):
        self._pipeline = pipeline_kwargs['pipeline']
        self._pipeline_config = pipeline_kwargs['pipeline_config']
        self._optimizer = pipeline_kwargs['optimizer']
        self._resource_scheduler = ['resource_scheduler']
        self._backend = ['backend']

    @typing.no_type_check
    def search(
        self,
        X,
        y,
        X_val,
        y_val,
        X_test,
        y_test,
        search_space,
    ):
        pass

    @typing.no_type_check
    def fit(
        self,
        X,
        y,
        X_test,
        y_test,
        model_config,
    ):
        self._pipeline.fit(X, y)

    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Generate the estimator predictions.

        Generate the predictions based on the given examples from the test set.

        Args:
        X_test: (np.ndarray)
            The test set examples.

        Returns:
            Array with estimator predictions.
        """
        #TODO discuss if we should pass the pipeline batch size here
        #TODO discuss if probabilities should be returned, maybe
        # that functionality should be included in a tabular task probably.
        return self._pipeline.predict(X_test)

    def score(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate the score on the test set.

        Calculate the evaluation measure on the test set.

        Args:
        X_test: (np.ndarray)
            The test examples of the dataset.
        y_test: (np.ndarray)
            The test ground truth labels.
        sample_weights: (np.ndarray|None)
            The weights for each sample.
        Returns:
            Value of the evaluation metric calculated on the test set.
        """
        return self._pipeline.score(X_test, y_test, sample_weights)

    def get_pipeline_config(self) -> Configuration:
        """Get the pipeline configuration.
        Returns:
            A Configuration object which is used to configure the pipeline.
        """
        return self._pipeline_config

    def set_pipeline_config(
        self,
        new_pipeline_config: Configuration,
    ):
        """Sets a new pipeline configuration.

        Args:
        new_pipeline_config (Configuration):
            The new pipeline configuration.
        """
        self._pipeline_config = new_pipeline_config

    @typing.no_type_check
    def get_incumbent_results(
        self
    ):
        pass

    @typing.no_type_check
    def get_incumbent_config(
        self
    ):
        pass

    def get_default_search_space(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ConfigurationSpace:
        """
        Args:
            X_train: (np.ndarray)
                The training examples of the dataset being used.
            y_train: (np.ndarray)
                The training labels of the dataset.
        Returns:
            The config space with the default hyperparameters.
        """
        # TODO discuss if pipeline_config neeeds to be included, since this is the default_search_space.
        # TODO discuss if X_val, y_val are needed. They were used in the old autopytorch.
        # get_dataset_info is a placeholder for the real function.
        dataset_properties = get_dataset_info(
            X_train,
            y_train,
            self.pipeline_config,
        )

        return self._pipeline.get_hyperparameter_search_space(
            dataset_properties,
        )
