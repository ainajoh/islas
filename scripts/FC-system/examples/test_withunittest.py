import numpy as np
from loclib.domain import *  # require netcdf4
from loclib.get_data import *
import unittest
from loclib.check_data import *  # require netcdf4

class TestThredds(unittest.TestCase):

    #def setUpClass(cls):
    #    #Useful to fex set up a database for all ur testeing..
    #    print("setUpClass")
    #def tarDownClass(cls):
    #    print("tarDownClass")

    def setUp(self):

        #runs before every test
        pass
    def tearDown(self):
        #runs after every test
        #you could delete all created files forexample.
        pass
    def test_domain(self):
        #all tests have to start with a test_ in func name.
        #all test are isolated and not in any rekkefolge. So be careful. You can though create a hierachy of test with class methods.
        #should not rely on other tests
        pass
    def test_get_data_init(self):
        pass
    def test_check_data(self):
        import numpy as np

        modelruntime = ["2020030800" ]
        YYYY = ["2016","2017","2018","2019"]
        MM = [["12"], ]
        model = "MEPS"
        data_domain = DOMAIN(modelruntime, model)
        data_domain.South_Norway()
        lonlat = np.array(data_domain.lonlat)
        param = ["surface_aerosol_sea"]
        levtype = "ml"
        lt = 1
        mbrs=0
        model = "MEPS"
        files = check_available(date=modelruntime, mbrs=mbrs, levtype=levtype, param = param, model=model)



    #def test_monthly_scedule(self):
    #    #with patch("employee.requests.get") as mocked_get: #incase webpage is down.. not ur fault
    #    #    mocked_get.return_value.ok = True
    #    #    mocked_get.return_value.text = "Success"##
    #
    #        sceduele = self.emp_1.monthly_scedule("May")
    #        mocked_get.assert_called_with("hhtp://company/Schafer/May")
    #        self.assertEqual(sceduele, "Success")#
    #
    #        #test bad result
    #        mocked_get.return_value.ok = False
    #        sceduele = self.emp_2.monthly_scedule("June")
    #        mocked_get.assert_called_with("hhtp://company/Smith/June")
    #        self.assertEqual(sceduele, "Bad Response!")


if __name__ =='__main__':
    unittest.main()

