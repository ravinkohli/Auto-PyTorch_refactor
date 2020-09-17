import copy
import unittest
import unittest.mock

import numpy as np

import torch

from autoPyTorch.pipeline.components.training.data_loader.base_data_loader import (
    BaseDataLoaderComponent
)


class BaseDataLoaderTest(unittest.TestCase):
    def test_get_set_config_space(self):
        """
        Makes sure that the configuration space of the base data loader
        is properly working"""
        loader = BaseDataLoaderComponent()

        cs = loader.get_hyperparameter_search_space()

        # Make sure that the batch size is a valid hyperparameter
        self.assertEqual(cs.get_hyperparameter('batch_size').default_value, 64)

        # Make sure we can properly set some random configs
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            loader.set_hyperparameters(config)

            self.assertEqual(loader.batch_size,
                             config_dict['batch_size'])

    def test_check_requirements(self):
        """ Makes sure that we catch the proper requirements for the
        data loader"""

        fit_dictionary = {}

        loader = BaseDataLoaderComponent()

        # Make sure we catch all possible errors in check requirements

        # No input in fit dictionary
        with self.assertRaisesRegex(ValueError,
                                    'Data loader requires the user to provide the input data'):
            loader.fit(fit_dictionary)

        # Wrong dataset
        fit_dictionary.update({'dataset': 'wrong'})
        with self.assertRaisesRegex(ValueError,
                                    'Unsupported dataset'):
            loader.fit(fit_dictionary)
        fit_dictionary['dataset'] = 'CIFAR10'
        with self.assertRaisesRegex(ValueError,
                                    'DataLoader needs the root of where'):
            loader.fit(fit_dictionary)
        fit_dictionary.pop('dataset')

        # X,y testing
        fit_dictionary.update({'X_train': unittest.mock.Mock()})
        with self.assertRaisesRegex(ValueError,
                                    'Data loader cannot access the train features-targets'):
            loader.fit(fit_dictionary)
        fit_dictionary.update({'y_train': unittest.mock.Mock()})
        with self.assertRaisesRegex(ValueError,
                                    'Data loader cannot access the indices needed to'):
            loader.fit(fit_dictionary)

    def test_fit_transform(self):
        """ Makes sure that fit and transform work as intended """
        fit_dictionary = {
            'X_train': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            'y_train': np.array([0, 1, 0]),
            'train_indices': [0, 1],
            'val_indices': [2],
        }

        # Mock child classes requirements
        loader = BaseDataLoaderComponent()
        loader.build_transform = unittest.mock.Mock()
        loader._check_transform_requirements = unittest.mock.Mock()

        loader.fit(fit_dictionary)

        # Fit means that we created the data loaders
        self.assertIsInstance(loader.train_data_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(loader.val_data_loader, torch.utils.data.DataLoader)

        # Transforms adds this fit dictionaries
        transformed_fit_dictionary = loader.transform(fit_dictionary)
        self.assertIn('train_data_loader', transformed_fit_dictionary)
        self.assertIn('val_data_loader', transformed_fit_dictionary)

        self.assertEqual(transformed_fit_dictionary['train_data_loader'],
                         loader.train_data_loader)
        self.assertEqual(transformed_fit_dictionary['val_data_loader'],
                         loader.val_data_loader)


if __name__ == '__main__':
    unittest.main()
