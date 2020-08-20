from typing import Optional

import ConfigSpace as cs


class SearchSpace:

    hyperparameter_types = {
        'categorical': cs.CategoricalHyperparameter,
        'integer': cs.UniformIntegerHyperparameter,
        'float': cs.UniformFloatHyperparameter,
        'constant': cs.Constant,
    }

    def __init__(
            self,
            cs_name: str = 'Default Hyperparameter Config',
            seed: int = 11,
    ):
        """Fit the selected algorithm to the training data.

        Args:
            cs_name (str): The name of the configuration space.
            seed (int): Seed value used for the configuration space.

        Returns:
        """
        self._hp_search_space = cs.ConfigurationSpace(
            name=cs_name,
            seed=seed,
        )

    def add_hyperparameter(
        self,
        name: str,
        hyperparameter_type: str,
        **kwargs,
    ):
        """Add a new hyperparameter to the configuration space.

        Args:
            name (str): The name of the hyperparameter to be added.
            type (str): The type of the hyperparameter to be added.

        Returns:
        """
        missing_arg = SearchSpace._assert_necessary_arguments_given(
            hyperparameter_type,
            **kwargs,
        )
        if missing_arg is not None:
            raise TypeError(f'A {hyperparameter_type} must have a value for {missing_arg}')
        else:
            self._hp_search_space.add_hyperparameter(
                SearchSpace.hyperparameter_types[type](
                    name=name,
                    **kwargs,
                )
            )

    @classmethod
    def _assert_necessary_arguments_given(
        hyperparameter_type: str,
        **kwargs,
    ) -> Optional[str]:
        """Assert that given a particular hyperparameter type, all the
        necessary arguments are given to create the hyperparameter.

        Args:
            hyperparameter_type (str): The type of the hyperparameter to be added.

        Returns:
            missing_argument (str|None): The argument that is missing
                to create the given hyperparameter.
        """
        necessary_args = {
            'categorical': {'choices', 'default_value'},
            'integer': {'lower', 'upper', 'default', 'log'},
            'float': {'lower', 'upper', 'default', 'log'},
            'constant': {'value'},
        }

        hp_necessary_args = necessary_args[hyperparameter_type]
        for hp_necessary_arg in hp_necessary_args:
            if hp_necessary_arg not in kwargs:
                return hp_necessary_arg

        return None

    def add_condition(
        self,
        hp1,
        hp2,
        cond,
    ):
        pass
