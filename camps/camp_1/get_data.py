#GET ONLY THE SPECIFIC DATA SPECIFIED IN MAIN SCRIPT
from netCDF4 import Dataset                     #For reading netcdf files.
import os
path = os.path.abspath("islas/camps/camp_1")
from netCDF4 import Dataset
import numpy as np

def SomeError( exception = Exception, message = "Something don't well" ):
    #source: https://softwareengineering.stackexchange.com/questions/222586/how-should-you-cleanly-restrict-object-property-types-and-values-in-python
    if isinstance( exception.args, tuple ):
        raise exception
    else:
        raise exception(message)

model = ["AromeArctic"] #ECMWF later
source = ["thredds"] # later"netcdf", "grib" 2019120100
type = ["full","sfx","ml"] #include pl here aswell.
filter_function_for_type= lambda value: value if value in type else SomeError(ValueError, f'Type not found: choices:{type}')

filter_function_for_models = lambda value: value if value in model else SomeError(ValueError, f'Model not found: choiced:{model}')
filter_function_for_source = lambda value: value if value in source else SomeError(ValueError, f'Source not found: choiced:{source}')
filter_function_for_modelrun = lambda value: value \
    if ( len(value) == 10 ) and ( int(value[0:4]) in range(2000,2021) ) and ( int(value[4:6]) in range(0,13) ) \
    and ( int(value[6:8]) in range(1,32)) and ( int(value[9:10]) in range(0,25)) \
    else SomeError(ValueError, f'Modelrun wrong: Either; "latest or date on the form: YYYYMMDDHH')


#class CALCULATE(DATA):
#
#    def pottemp(self):
#        theta = self.air_temperature_ml * (p0 / p) ** 0.286
#        self.theta = theta#
#


class DATA():
    def __init__(self, data_domain, model="AromeArctic", type ="full", source="thredds", modelrun="latest", fctime = [0,66],height_ml = [0,64], param_ML = None, param_SFC = None, param_sfx = None):

        self.data_domain = data_domain
        self.source = source
        self.type = type
        self.model = model
        self.modelrun = modelrun
        self.fctime = fctime
        self.height_ml =  height_ml
        self.param_ML = param_ML
        self.param_SFC = param_SFC
        self.param_sfx = param_sfx

    def __setattr__(self, key, value):

        if key=='type':
            value = filter_function_for_type(value)
            if value =="ml":
                value = "full"
            self.__dict__[key] = value
        if key=='model':
            value = filter_function_for_models(value)
            self.__dict__[key] = value
        if key=='source':
            value = filter_function_for_source(value)
            self.__dict__[key] = value
        if key=='modelrun':
            if value !="latest":
                value = filter_function_for_modelrun(value)
            self.__dict__[key] = value
        if key=="data_domain": #or else obj would not be properly set...
            self.__dict__[key] = value
        if key == "param_ML":  # or else obj would not be properly set...
            self.__dict__[key] = value
        if key == "param_SFC":  # or else obj would not be properly set...
            self.__dict__[key] = value
        if key == "param_sfx":  # or else obj would not be properly set...
            self.__dict__[key] = value
        if key == "fctime":  # or else obj would not be properly set...
            self.__dict__[key] = value
        if key == "height_ml":  # or else obj would not be properly set...
            self.__dict__[key] = value

    def make_url(self):
        jindx = self.data_domain.idx[0]
        iindx = self.data_domain.idx[1]
        if self.modelrun != "latest":
            YYYY=self.modelrun[0:4]
            MM = self.modelrun[4:6]
            DD = self.modelrun[6:8]
            HH = self.modelrun[9:10]
            url = f"https://thredds.met.no/thredds/dodsC/{YYYY}/{MM}/{DD}/arome_arctic_{self.type}_2_5km_{YYYY}{MM}{DD}T{HH}Z.nc"
        else:
            url = f"https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_{self.type}_2_5km_latest.nc"

        url += f"?time[{np.min(self.fctime)}:1:{np.max(self.fctime)}]," + \
               f"latitude[{jindx.min()}:1:{jindx.max()}][{iindx.min()}:1:{iindx.max()}]," + \
               f"longitude[{jindx.min()}:1:{jindx.max()}][{iindx.min()}:1:{iindx.max()}]"

        if self.type == "full":
            url += f",hybrid[{np.min(self.height_ml)}:1:{np.max(self.height_ml)}]," + \
                   f"ap[{np.min(self.height_ml)}:1:{np.max(self.height_ml)}]," + \
                   f"b[{np.min(self.height_ml)}:1:{np.max(self.height_ml)}]"
        if self.param_ML:
            for prm in self.param_ML:
                url +=f",{prm}[{np.min(self.fctime)}:1:{np.max(self.fctime)}][{np.min(self.height_ml)}:1:{np.max(self.height_ml)}][{jindx.min()}:1:{jindx.max()}][{iindx.min()}:1:{iindx.max()}]"
        if self.param_SFC:
            for prm in self.param_SFC:
                url += f",{prm}[{np.min(self.fctime)}:1:{np.max(self.fctime)}][0][{jindx.min()}:1:{jindx.max()}][{iindx.min()}:1:{iindx.max()}]"
        if self.param_sfx:
            for prm in self.param_sfx:
                url += f",{prm}[{np.min(self.fctime)}:1:{np.max(self.fctime)}][{jindx.min()}:1:{jindx.max()}][{iindx.min()}:1:{iindx.max()}]"


        self.__dict__["url"] = url
        #print(url)
        return url

    def thredds(self, url):
        print("-------> start retrieve from thredds")
        dataset = Dataset(url) #fast
        print("-------> Getting variable: ")

        for prm in ["time", "latitude", "longitude"]:
            print(prm)
            self.__dict__[prm] = dataset.variables[prm][:]
        if self.type =="full":
            for prm in ["hybrid", "ap", "b" ]:
                print(prm)
                self.__dict__[prm] = dataset.variables[prm][:]
        if self.param_ML:
            for prm in self.param_ML:
                print(prm)
                self.__dict__[prm] = dataset.variables[prm][:]
        if self.param_SFC:
            for prm in self.param_SFC:
                print(prm)
                self.__dict__[prm] = dataset.variables[prm][:]
        if self.param_sfx:
            for prm in self.param_sfx:
                print(prm)
                self.__dict__[prm] = dataset.variables[prm][:]
        dataset.close()


    def retrieve(self):
        if self.source == "thredds":
            url = self.make_url()
            self.thredds(url)

