import unittest
import unittest.mock

from sklearn.datasets import make_classification

from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


class PipelineTest(unittest.TestCase):

    def setUp(self):
        self.num_features = 4
        self.num_classes = 2
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=self.num_features,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=self.num_classes,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0
        )
        self.dataset_properties = {
            'task_type': 'tabular_classification',
            'output_type': 'binary',
            'numerical_columns': list(range(4)),
            'categorical_columns': [],
        }

    def test_pipeline_fit(self):
        """This test makes sure that the pipeline is able to fit
        given random combinations of hyperparameters across the pipeline"""

        pipeline = TabularClassificationPipeline(dataset_properties=self.dataset_properties)
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)
        print(config)
        pipeline.fit(
            {'num_features': self.num_features,
             'num_classes': self.num_classes,
             'numerical_columns': list(range(self.num_features)),
             'categorical_columns': [],
             'categories': [],
             'X_train': self.X,
             'y_train': self.y,
             'train_indices': range(self.X.shape[0] // 2),
             'val_indices': range(self.X.shape[0] // 2, self.X.shape[0]),
             'is_small_preprocess': False,
             # Training configuration
             'dataset_properties': self.dataset_properties,
             'job_id': 'example_tabular_classification_1',
             'device': 'cpu',
             'budget_type': 'epochs',
             'epochs': 5,
             'torch_num_threads': 1,
             'early_stopping': 20,
             'working_dir': '/tmp',
             'use_tensorboard_logger': True,
             'use_pynisher': False,
             'metrics_during_training': True,
             }
        )

        # To make sure we fitted the model, there should be a
        # run summary object with accuracy
        self.assertIsNotNone(pipeline.named_steps['trainer'].run_summary)

    def test_default_configuration(self):
        """Makes sure that when no config is set, we can trust the
        default configuration from the space"""
        pipeline = TabularClassificationPipeline(dataset_properties=self.dataset_properties)

        pipeline.fit(
            {'num_features': self.num_features,
             'num_classes': self.num_classes,
             'numerical_columns': list(range(self.num_features)),
             'categorical_columns': [],
             'categories': [],
             'X_train': self.X,
             'y_train': self.y,
             'train_indices': range(self.X.shape[0] // 2),
             'val_indices': range(self.X.shape[0] // 2, self.X.shape[0]),
             'is_small_preprocess': False,
             # Training configuration
             'dataset_properties': self.dataset_properties,
             'job_id': 'example_tabular_classification_1',
             'device': 'cpu',
             'budget_type': 'epochs',
             'epochs': 5,
             'torch_num_threads': 1,
             'early_stopping': 20,
             'working_dir': '/tmp',
             'use_tensorboard_logger': True,
             'use_pynisher': False,
             'metrics_during_training': True,
             }
        )

    def test_network_optimizer_lr_handshake(self):
        """Fitting a network should put the network in the X"""

        # Create the pipeline to check. A random config should be sufficient
        dataset_properties = {'numerical_columns': [], 'categorical_columns': []}
        pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)
        cs = pipeline.get_hyperparameter_search_space()
        config = cs.sample_configuration()
        pipeline.set_hyperparameters(config)

        # Make sure that fitting a network adds a "network" to X
        self.assertIn('network', pipeline.named_steps.keys())
        X = pipeline.named_steps['network'].fit(
            {'num_features': 10, 'num_classes': 2},
            None
        ).transform({})
        self.assertIn('network', X)

        # Then fitting a optimizer should fail if no network:
        self.assertIn('optimizer', pipeline.named_steps.keys())
        with self.assertRaisesRegex(ValueError, 'Could not parse the network'):
            pipeline.named_steps['optimizer'].fit({}, None)

        # No error when network is passed
        X = pipeline.named_steps['optimizer'].fit(X, None).transform(X)
        self.assertIn('optimizer', X)

        # Then fitting a optimizer should fail if no network:
        self.assertIn('lr_scheduler', pipeline.named_steps.keys())
        with self.assertRaisesRegex(ValueError,
                                    'the fit dictionary Must contain a valid optimizer'):
            pipeline.named_steps['lr_scheduler'].fit({}, None)

        # No error when network is passed
        X = pipeline.named_steps['lr_scheduler'].fit(X, None).transform(X)
        self.assertIn('optimizer', X)


if __name__ == '__main__':
    unittest.main()
