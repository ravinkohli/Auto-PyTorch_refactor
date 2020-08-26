import unittest
import pandas as pd
import numpy as np
from autoPyTorch.datasets.tabular_dataset import DataTypes, TabularDataset


class DataFrameTest(unittest.TestCase):
    def runTest(self):
        df = pd.DataFrame([['a', 0.1, 1], ['b', 0.2, np.nan]])
        target_df = pd.Series([1, 2])
        ds = TabularDataset(df, target_df)
        self.assertEqual(ds.data_types, [DataTypes.String, DataTypes.Float, DataTypes.Canonical])
        self.assertEqual(set(ds.itovs[2]), {np.nan, 1})
        self.assertEqual(set(ds.itovs[0]), {np.nan, 'a', 'b'})

        self.assertEqual(ds.vtois[0]['a'], 2)
        self.assertEqual(ds.vtois[0][np.nan], 0)
        self.assertEqual(ds.vtois[0][pd._libs.NaT], 0)
        self.assertEqual(ds.vtois[0][pd._libs.missing.NAType()], 0)
        self.assertTrue((ds.nan_mask == np.array([[0, 0, 0], [0, 0, 1]], dtype=np.bool)).all())


class NumpyArrayTest(unittest.TestCase):
    def runTest(self):
        matrix = np.array([(0, 0.1, 1), (1, np.nan, 3)], dtype='f4, f4, i4')
        target_df = pd.Series([1, 2])
        ds = TabularDataset(matrix, target_df)
        self.assertEqual(ds.data_types, [DataTypes.Canonical, DataTypes.Float, DataTypes.Canonical])
        self.assertEqual(set(ds.itovs[2]), {np.nan, 1, 3})

        self.assertEqual(ds.vtois[0][1], 2)
        self.assertEqual(ds.vtois[0][np.nan], 0)
        self.assertEqual(ds.vtois[0][pd._libs.NaT], 0)
        self.assertEqual(ds.vtois[0][pd._libs.missing.NAType()], 0)
        self.assertTrue((ds.nan_mask == np.array([[0, 0, 0], [0, 1, 0]], dtype=np.bool)).all())


if __name__ == '__main__':
    unittest.main()
