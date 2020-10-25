import unittest

from imgaug.augmenters.meta import Augmenter, Sequential

import numpy as np

from autoPyTorch.pipeline.components.setup.augmentation.image.ImageAugmenter import ImageAugmenter


class TestImageAugmenter(unittest.TestCase):
    def test_every_augmenter(self):
        image_augmenter = ImageAugmenter()
        configuration = image_augmenter.get_hyperparameter_search_space().sample_configuration()
        image_augmenter = image_augmenter.set_hyperparameters(configuration=configuration)
        for name, augmenter in image_augmenter.available_augmenters.items():
            X = dict(X_train=np.random.randint(0, 255, (8, 3, 16, 16), dtype=np.uint8))
            # check if correct augmenter is saved in available augmenters
            self.assertEqual(name, image_augmenter.available_augmenters[name].__class__.__name__)

            augmenter = augmenter.fit(X)
            # test if augmenter has an Augmenter attribute
            self.assertIsInstance(augmenter.get_image_augmenter(), Augmenter)

            # test if augmenter works on a random image
            train_aug = augmenter(X['X_train'])
            self.assertIsInstance(train_aug, np.ndarray)

    def test_get_set_config_space(self):
        X = dict(X_train=np.random.randint(0, 255, (8, 3, 16, 16), dtype=np.uint8))
        image_augmenter = ImageAugmenter()
        configuration = image_augmenter.get_hyperparameter_search_space().sample_configuration()
        image_augmenter = image_augmenter.set_hyperparameters(configuration=configuration)
        image_augmenter = image_augmenter.fit(X)
        X = image_augmenter.transform(X)

        image_augmenter = X['image_augmenter']
        # test if a sequential augmenter was formed
        self.assertIsInstance(image_augmenter.augmenter, Sequential)

        # test if augmenter works on a random image
        train_aug = image_augmenter(X['X_train'])
        self.assertIsInstance(train_aug, np.ndarray)


if __name__ == '__main__':
    unittest.main()
