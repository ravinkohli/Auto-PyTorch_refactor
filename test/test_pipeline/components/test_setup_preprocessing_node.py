import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torchvision.transforms

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.padding.base_pad_choice import PadChoice
from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.normalise.base_normalizer_choice import (
    NormalizerChoice
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder_choice import EncoderChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.SimpleImputer import SimpleImputer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler_choice import ScalerChoice
from autoPyTorch.pipeline.components.setup.preprocessor.Preprocessing import Preprocessing
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


class TabularPreprocessingTest(unittest.TestCase):
    def test_tabular_preprocess(self):
        X = dict(train=np.random.random((10, 15)),
                 is_small_preprocess=True,
                 numerical_columns=list(range(15)),
                 categorical_columns=[],
                 num_features=15,
                 num_classes=2
                 )
        dataset_properties = dict(numerical_columns=list(range(15)),categorical_columns=[],)
        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertNotIn('preprocess_transforms', X)
        self.assertNotIn('imputer', X)
        self.assertNotIn('encoder', X)
        self.assertNotIn('scaler', X)

    def test_tabular_no_preprocess(self):
        X = dict(train=np.random.random((10, 15)),
                 is_small_preprocess=False,
                 numerical_columns=list(range(15)),
                 categorical_columns=[],
                 num_features=15,
                 num_classes=2
                 )
        dataset_properties = dict(numerical_columns=list(range(15)),categorical_columns=[],)

        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertIn('preprocess_transforms', X)
        self.assertIsInstance(X['preprocess_transforms'], torchvision.transforms.Compose)
        self.assertNotIn('imputer', X)
        self.assertNotIn('encoder', X)
        self.assertNotIn('scaler', X)


class ImagePreprocessingTest(unittest.TestCase):
    def test_image_preprocess(self):
        data = np.random.random((10, 2, 2, 3))
        X = dict(train=data,
                 is_small_preprocess=True,
                 channelwise_mean=np.array([np.mean(data[:, :, :, i]) for i in range(3)]),
                 channelwise_std=np.array([np.std(data[:, :, :, i]) for i in range(3)]),
                 )
        dataset_properties = dict()
        pipeline = ImageClassificationPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertNotIn('preprocess_transforms', X)
        self.assertNotIn('pad', X)
        self.assertNotIn('normalizer', X)

    def test_image_no_preprocess(self):
        data = np.random.random((10, 2, 2, 3))
        X = dict(train=data,
                 is_small_preprocess=False,
                 channelwise_mean=np.array([np.mean(data[:, :, :, i]) for i in range(3)]),
                 channelwise_std=np.array([np.std(data[:, :, :, i]) for i in range(3)]),
                 )
        dataset_properties = dict()
        pipeline = ImageClassificationPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertIn('preprocess_transforms', X)
        self.assertIsInstance(X['preprocess_transforms'], torchvision.transforms.Compose)
        self.assertNotIn('pad', X)
        self.assertNotIn('normalizer', X)


if __name__ == '__main__':
    unittest.main()
