import numpy as np
import unittest
from imetkit.check_data import check_data
from imetkit.get_data import *

# MethodName_StateUnderTest_ExpectedBehavior


class TestThredds(unittest.TestCase):
    # def setUpClass(cls):
    #    #Useful to fex set up a database for all ur testeing..
    #    print("setUpClass")
    # def tarDownClass(cls):
    #    print("tarDownClass")
    def setUp(self):
        check_ml = check_data()
        check_pl = check_data()

    def tearDown(self):
        # remove logging perhaps
        # runs after every test
        # you could delete all created files forexample.
        pass

    def test_ModelParam_OnlyModelInput_ShowMostUsedParam(self):
        pass


if __name__ == "__main__":
    unittest.main()
