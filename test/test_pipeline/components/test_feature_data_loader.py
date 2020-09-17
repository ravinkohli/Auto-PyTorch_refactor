import unittest
import unittest.mock

import torchvision

from autoPyTorch.pipeline.components.training.data_loader.feature_data_loader import (
    FeatureDataLoader
)


class TestFeatureDataLoader(unittest.TestCase):
    def test_build_transform(self):
        """
        Makes sure a proper composition is created
        """
        loader = FeatureDataLoader()

        fit_dictionary = {}
        for thing in ['imputer', 'scaler', 'encoder']:
            fit_dictionary[thing] = unittest.mock.Mock()

        compose = loader.build_transform(fit_dictionary)

        self.assertIsInstance(compose, torchvision.transforms.Compose)

        # Make sure 4 transformations where created
        # imputer, scaler, encoder and to tensor
        self.assertEqual(len(compose.transforms), 4)
