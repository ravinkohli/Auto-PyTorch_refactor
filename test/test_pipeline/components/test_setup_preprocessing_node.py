import unittest

import numpy as np

import torchvision.transforms

from autoPyTorch.pipeline.image_classification import ImageClassificationPipeline
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


class TabularPreprocessingTest(unittest.TestCase):
    def test_tabular_preprocess(self):
        dataset_properties = dict(numerical_columns=list(range(15)), categorical_columns=[],)
        X = dict(X_train=np.random.random((10, 15)),
                 y_train=np.random.random(10),
                 train_indices=[0, 1, 2, 3, 4, 5],
                 val_indices=[6, 7, 8, 9],
                 is_small_preprocess=True,
                 numerical_columns=list(range(15)),
                 categorical_columns=[],
                 num_features=15,
                 num_classes=2,
                 categories=[],
                 # Training configuration
                 job_id='test',
                 device='cpu',
                 budget_type='epochs',
                 epochs=10,
                 torch_num_threads=1,
                 early_stopping=20,
                 dataset_properties=dataset_properties,
                 )
        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        # Remove the trainer
        pipeline.steps.pop()
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertNotIn('preprocess_transforms', X.keys())

    def test_tabular_no_preprocess(self):
        dataset_properties = dict(numerical_columns=list(range(15)), categorical_columns=[],
                                  task_type='tabular_classification', output_type='binary')
        X = dict(X_train=np.random.random((10, 15)),
                 y_train=np.random.random(10),
                 train_indices=[0, 1, 2, 3, 4, 5],
                 val_indices=[6, 7, 8, 9],
                 is_small_preprocess=False,
                 numerical_columns=list(range(15)),
                 categorical_columns=[],
                 num_features=15,
                 num_classes=2,
                 categories=[],
                 # Training configuration
                 job_id='test',
                 device='cpu',
                 budget_type='epochs',
                 epochs=10,
                 torch_num_threads=1,
                 early_stopping=20,
                 dataset_properties=dataset_properties,
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
        X = dict(X_train=data,
                 y_train=np.random.random(10),
                 train_indices=[0, 1, 2, 3, 4, 5],
                 val_indices=[6, 7, 8, 9],
                 is_small_preprocess=True,
                 channelwise_mean=np.array([np.mean(data[:, :, :, i]) for i in range(3)]),
                 channelwise_std=np.array([np.std(data[:, :, :, i]) for i in range(3)]),
                 )
        dataset_properties = dict()
        pipeline = ImageClassificationPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertNotIn('preprocess_transforms', X.keys())

    def test_image_no_preprocess(self):
        data = np.random.random((10, 2, 2, 3))
        X = dict(X_train=data,
                 y_train=np.random.random(10),
                 train_indices=[0, 1, 2, 3, 4, 5],
                 val_indices=[6, 7, 8, 9],
                 is_small_preprocess=False,
                 channelwise_mean=np.array([np.mean(data[:, :, :, i]) for i in range(3)]),
                 channelwise_std=np.array([np.std(data[:, :, :, i]) for i in range(3)]),
                 )
        dataset_properties = dict()
        pipeline = ImageClassificationPipeline(dataset_properties=dataset_properties)
        pipeline = pipeline.fit(X)
        X = pipeline.transform(X)
        self.assertIn('preprocess_transforms', X.keys())
        self.assertIsInstance(X['preprocess_transforms'], torchvision.transforms.Compose)


if __name__ == '__main__':
    unittest.main()
