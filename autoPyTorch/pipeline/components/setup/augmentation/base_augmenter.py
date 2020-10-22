from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class BaseAugmenter(autoPyTorchComponent):

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted augmenter into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        raise NotImplementedError()

    #
    # def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
    #     """
    #     A mechanism in code to ensure the correctness of the fit dictionary
    #     It recursively makes sure that the children and parent level requirements
    #     are honored before fit.
    #
    #     Args:
    #         X (Dict[str, Any]): Dictionary with fitted parameters. It is a message passing
    #             mechanism, in which during a transform, a components adds relevant information
    #             so that further stages can be properly fitted
    #     """
    #     super().check_requirements(X, y)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, str]] = None
    ) -> ConfigurationSpace:
        """Return the configuration space of this algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]): Describes the dataset
               to work on

        Returns:
            ConfigurationSpace: The configuration space of this algorithm.
        """
        return ConfigurationSpace()