import unittest

from rPTMDetermine.features import Features


class TestFeatures(unittest.TestCase):
    def test_uninitialized(self):
        features = Features()

        self.assertEqual([], features.feature_names())
        self.assertEqual([], features.to_list())

    def test_initialization(self):
        features = Features()

        features.set('NumPeaks', 2.)
        features.set('PepLen', 9.)

        self.assertEqual(2., features.get('NumPeaks'))
        self.assertEqual(9., features.get('PepLen'))
        self.assertEqual(['NumPeaks', 'PepLen'], features.feature_names())
        self.assertEqual([2., 9.], features.to_list())
        self.assertEqual([9.], features.to_list(['PepLen']))


if __name__ == '__main__':
    unittest.main()
