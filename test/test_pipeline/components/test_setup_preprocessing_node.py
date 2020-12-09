import unittest
from unittest import mock

import numpy as np

import torchvision.transforms

from autoPyTorch.constants import (
    TABULAR_CLASSIFICATION,
    TASK_TYPES_TO_STRING,
    MULTICLASS,
    OUTPUT_TYPES_TO_STRING
)
from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


class TabularPreprocessingTest(unittest.TestCase):
    def setUp(self):
        # Setup the backed for this test
        self.backend = mock.Mock()
        dataset = mock.MagicMock()
        dataset.__len__.return_value = 1
        datamanager = mock.MagicMock()
        datamanager.get_dataset_for_training.return_value = (dataset, dataset)
        self.backend.load_datamanager.return_value = datamanager

    def test_tabular_preprocess(self):
        dataset_properties = {
            'numerical_columns': list(range(15)),
            'categorical_columns': [],
            'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
            'output_type': OUTPUT_TYPES_TO_STRING[MULTICLASS],
            'is_small_preprocess': True,
            'num_features': 15,
            'num_classes': 2,
            'categories': [],
        }
        X = dict(X_train=np.random.random((10, 15)),
                 y_train=np.random.random(10),
                 train_indices=[0, 1, 2, 3, 4, 5],
                 val_indices=[6, 7, 8, 9],
                 dataset_properties=dataset_properties,
                 # Training configuration
                 job_id='test',
                 device='cpu',
                 budget_type='epochs',
                 epochs=10,
                 torch_num_threads=1,
                 early_stopping=20,
                 split_id=0,
                 backend=self.backend,
                 )
        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        # Remove the trainer
        pipeline.steps.pop()
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertNotIn('preprocess_transforms', X.keys())

    def test_tabular_no_preprocess(self):
        dataset_properties = {
            'numerical_columns': list(range(15)),
            'categorical_columns': [],
            'task_type': TASK_TYPES_TO_STRING[TABULAR_CLASSIFICATION],
            'output_type': OUTPUT_TYPES_TO_STRING[MULTICLASS],
            'is_small_preprocess': False,
            'num_features': 15,
            'num_classes': 2,
            'categories': [],
        }
        X = dict(X_train=np.random.random((10, 15)),
                 y_train=np.random.random(10),
                 train_indices=[0, 1, 2, 3, 4, 5],
                 val_indices=[6, 7, 8, 9],
                 dataset_properties=dataset_properties,
                 # Training configuration
                 job_id='test',
                 device='cpu',
                 budget_type='epochs',
                 epochs=10,
                 torch_num_threads=1,
                 early_stopping=20,
                 split_id=0,
                 backend=self.backend,
                 )

        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        # Remove the trainer
        pipeline.steps.pop()
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertIn('preprocess_transforms', X.keys())
        self.assertIsInstance(X['preprocess_transforms'], torchvision.transforms.Compose)


class ImagePreprocessingTest(unittest.TestCase):
    def test_image_preprocess(self):
        data = np.random.random((10, 2, 2, 3))
        dataset_properties = dict(image_height=2,
                                  image_width=2,
                                  is_small_preprocess=True,
                                  mean=np.array([np.mean(data[:, :, :, i]) for i in range(3)]),
                                  std=np.array([np.std(data[:, :, :, i]) for i in range(3)]),
                                  )
        X = dict(X_train=data,
                 y_train=np.random.random(10),
                 train_indices=[0, 1, 2, 3, 4, 5],
                 val_indices=[6, 7, 8, 9],
                 dataset_properties=dataset_properties
                 )

        pipeline = ImageClassificationPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertNotIn('preprocess_transforms', X.keys())

    def test_image_no_preprocess(self):
        data = np.random.random((10, 2, 2, 3))
        dataset_properties = dict(image_height=2,
                                  image_width=2,
                                  is_small_preprocess=False,
                                  mean=np.array([np.mean(data[:, :, :, i]) for i in range(3)]),
                                  std=np.array([np.std(data[:, :, :, i]) for i in range(3)]),
                                  )
        X = dict(X_train=data,
                 y_train=np.random.random(10),
                 train_indices=[0, 1, 2, 3, 4, 5],
                 val_indices=[6, 7, 8, 9],
                 dataset_properties=dataset_properties
                 )
        dataset_properties = dict()
        pipeline = ImageClassificationPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertIn('preprocess_transforms', X.keys())
        self.assertIsInstance(X['preprocess_transforms'], torchvision.transforms.Compose)


if __name__ == '__main__':
    unittest.main()
