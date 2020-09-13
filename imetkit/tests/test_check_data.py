import numpy as np
import unittest
from FCsystem.check_data import check_data
from datetime import timedelta
from datetime import datetime



class TestThredds(unittest.TestCase):
    #def setUpClass(cls):
    #    #Useful to fex set up a database for all ur testeing..
    #    print("setUpClass")
    #def tarDownClass(cls):
    #    print("tarDownClass")
    def setUp(self):
        self.checkOnlyModelMEPS = check_data(model="MEPS",numbervar=5)
        self.checkOnlyModelAA = check_data(model="AromeArctic",numbervar=5)

        self.checkMEPSonDate = check_data(model="MEPS", date = "2020010100")
        self.checkAAonDate = check_data(model="AromeArctic", date = "2020010100")

        self.checkSearchMEPSonDate = check_data(model="MEPS", date="2020010100", search="temp")
        self.checkSearchAAonDate = check_data(model="AromeArctic", date="2020010100", search="temp")

        self.checkSearchMEPS = check_data(model="MEPS", search="wind")
        self.checkSearchAA = check_data(model="AromeArctic",search="wind")

        #runs before every test
        pass
    def tearDown(self):
        #runs after every test
        #you could delete all created files forexample.
        pass


    def test_available_variables_mostused(self):
        self.assertTrue(self.checkOnlyModelMEPS.param != None)
        self.assertTrue(type(self.checkOnlyModelMEPS.param)==str)

        self.assertTrue(self.checkOnlyModelAA.param != None)
        self.assertTrue(type(self.checkOnlyModelAA.param) == str)
        #One known output
        output ="""0                       time
1    forecast_reference_time
2         projection_lambert
3                          x
4                          y"""
        self.assertEqual(self.checkOnlyModelMEPS.param, output)
        self.assertEqual(self.checkOnlyModelAA.param, output)

    def test_available_variables_search(self):

        self.assertTrue(self.checkSearchAA.param != None)
        self.assertTrue(type(self.checkSearchAA.param) == str)
        self.assertTrue(self.checkSearchMEPS.param != None)
        self.assertTrue(type(self.checkSearchMEPS.param) == str)

        #One known output
        output = """0                             x_wind_10m
1                             y_wind_10m
2                        x_wind_gust_10m
3                              x_wind_ml
4                        y_wind_gust_10m
5                              y_wind_ml
6                     wind_speed_of_gust
7                              x_wind_pl
8                              y_wind_pl
9                               x_wind_z
10                              y_wind_z
11                        wind_direction
12                            wind_speed
13    atmosphere_level_of_max_wind_speed
14    x_wind_at_maximum_wind_speed_level
15    y_wind_at_maximum_wind_speed_level"""
        self.assertEqual(self.checkSearchAA.param, output)

    def test_available_variables_search4specdate(self):

        self.assertTrue(self.checkSearchMEPSonDate.param != None)
        self.assertTrue(type(self.checkSearchMEPSonDate.param) == str)
        self.assertTrue(self.checkSearchAAonDate.param != None)
        self.assertTrue(type(self.checkSearchAAonDate.param) == str)

        output ="""    arome_arctic_sfx_2_5km_20200101T00Z.nc arome_arctic_full_2_5km_20200101T00Z.nc arome_arctic_extracted_2_5km_20200101T00Z.nc
24                                     NaN                      air_temperature_0m                                          NaN
31                                     NaN                                     NaN                           air_temperature_0m
38                                     NaN                                     NaN                           air_temperature_ml
49                                     NaN                                     NaN                           air_temperature_pl
55                                     NaN                      air_temperature_2m                                          NaN
58                                     NaN                     air_temperature_min                            air_temperature_z
59                                     NaN                     air_temperature_max                                          NaN
65                                     NaN                      air_temperature_ml                                          NaN
75                                     NaN                                     NaN                           air_temperature_2m
97                                     NaN                                     NaN                          air_temperature_max
100                                    NaN                                     NaN                          air_temperature_min"""

        self.assertEqual(self.checkSearchAAonDate.param, output)

    def ifmodelisavailableevery6hours(self):
        pass
        #normalhours = ["00", "06", "12", "18"]
        #checkaa = check_data(model="AromeArctic")
        #modelrun = checkaa.date['Date'].astype(str) + checkaa.date['Hour'].astype(str).str.zfill(2)  # + str(check.date['Hour'])
        #startmodelrun = modelrun[0]

    def test_availabledate(self):
        self.assertEqual(self.checkOnlyModelAA.date.at[0,"Date"], 20151021)
        self.assertEqual(self.checkOnlyModelAA.date.at[0, "Hour"], 15)
        self.assertEqual(self.checkOnlyModelAA.date.at[14085,"Date"], 20200910)

        self.assertEqual(self.checkOnlyModelMEPS.date.at[0, "Date"], 20161108)
        self.assertEqual(self.checkOnlyModelMEPS.date.at[0, "Hour"], 0)
        self.assertEqual(self.checkOnlyModelMEPS.date.at[7613, "Date"], 20200505)

    def all_availablefiles(self): #timeconsuming, not in use
        # multiple
        models = ["MEPS"]  # , "AromeArctic"]
        for m in models:
            check = check_data(model=m)
            check.date = check.date[check.date['Hour'].isin(["00", "06", "12", "18"])]
            modelrun = check.date['Date'].astype(str) + check.date['Hour'].astype(str).str.zfill(2)  # + str(check.date['Hour'])
            for d in modelrun:
                check = check_data(model=m, date=d)
                self.assertTrue(len(check.file) > 0)
                self.assertTrue(type(check.file) == pd.core.frame.DataFrame)
        #        #print(check.date)

    def test_availablefiles(self):
        output="""0          arome_arctic_sfx_2_5km_20200101T00Z.nc
1         arome_arctic_full_2_5km_20200101T00Z.nc
2    arome_arctic_extracted_2_5km_20200101T00Z.nc"""
        self.assertEqual(self.checkAAonDate.file["File"].to_string(), output)

        output = """0               meps_sfx_2_5km_20200101T00Z.nc
1          meps_mbr0_sfx_2_5km_20200101T00Z.nc
2         meps_mbr0_full_2_5km_20200101T00Z.nc
3    meps_mbr0_extracted_2_5km_20200101T00Z.nc
4              meps_full_2_5km_20200101T00Z.nc
5         meps_extracted_2_5km_20200101T00Z.nc"""
        self.assertEqual(self.checkMEPSonDate.file["File"].to_string(), output)

    def test_availablefiles4setparameter(self):
        pass
    def test_availablefiles4setleveltype(self):
        pass

    def test_availablefiles4setensmember(self):
        pass

    # def test_availablefiles4seteverything(self):
        #check_data(model, date=None, param=None, mbrs=None, levtype=None, file=None, numbervar=100,
        #                search=None)
        #print(check.file)


    def test_check_data(self):
        pass
        #Find available variables
        # YYYY = ["2016","2017","2018","2019"]
        # MM = ["12"],
        #modelruntime = "2020030800"
        #param = ["surface_aerosol_sea"]
        #levtype = "ml"
        #mbrs=0
        #model = "MEPS"
        #files = check_data(date=modelruntime, mbrs=mbrs, levtype=levtype, param = param, model=model)



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

