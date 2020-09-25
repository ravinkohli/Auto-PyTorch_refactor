import unittest

import numpy as np
from numpy.testing import assert_array_equal

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.padding.NoPad import NoPad
from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.padding.Pad import Pad


class TestPad(unittest.TestCase):
    def initialise(self):
        self.train = np.random.randint(0, 255, (10, 3, 2, 2))

    def test_pad(self):
        self.initialise()
        mode_choices = Pad.get_hyperparameter_search_space().get_hyperparameter('mode')
        border = 2
        for mode in mode_choices.choices:
            pad = Pad(mode=mode)
            X = {'train': self.train}
            pad = pad.fit(X)
            X = pad.transform(X)

            # check if pad added to X is instance of self
            self.assertEqual(X['pad'], pad)
            assert_array_equal(np.pad(self.train, [(0, 0), (border, border), (border, border), (0, 0)], mode=mode),
                               pad(self.train))

    def test_no_pad(self):
        self.initialise()
        pad = NoPad()
        X = {'train': self.train}
        pad = pad.fit(X)
        X = pad.transform(X)

        # check if pad added to X is instance of self
        self.assertEqual(X['pad'], pad)

        assert_array_equal(self.train, pad(self.train))
