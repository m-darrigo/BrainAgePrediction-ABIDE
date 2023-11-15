import unittest
import os
import sys

#to run locally
#package_name = "../BrainAgePrediction"
package_name = "BrainAgePrediction"

sys.path.insert(0, package_name)

class TestBrainAge(unittest.TestCase):
    """
    Class for performing unit testing on code
    """
    def setUp(self):
        """
        Class setup.
        """
        self.data = package_name + "/dataset/FS_features_ABIDE_males.csv"

if __name__ == "__main__":
    unittest.main()
