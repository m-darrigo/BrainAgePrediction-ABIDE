""" Tests"""
import unittest
import numpy as np

import preprocessing_lib as lib


class TestPreprocessing(unittest.TestCase):
    def __init__(self):
        self.df = pd.read_csv(...)

    def test_alldistr_img(self):
        # This is a test to see if the function runs without errors.
        # The function is so simple that it does not need deep testing
        lib.alldistr_img(self.df)

    def test_corr_distr(self):
        # This is a test to see if the function runs without errors.
        # The function is so simple that it does not need deep testing
        lib.corr_distr(self.df)

    def test_scaled_distributions(self):
        # This is a test to see if the function runs without errors.
        # The function is so simple that it does not need deep testing
        lib.scaled_distributions(self.df)

    # test if the function returns a ValueError when a wrong input is given.
    def test_alldistr_img_wrong_dtype(self):
        with self.assertRaises(ValueError):
            lib.alldistr_img(5)

    # This type of testing can be utilized for all plotting functions


if __name__ == "__main__":
    unittest.main()
