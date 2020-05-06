"""
versions:
v1: AromeArcic on thredds on Svalbars.
v2: + MEPS at Finse with members
"""
from netCDF4 import Dataset
import numpy as np
import logging
import pandas as pd
from loclib.check_data import *  # require netcdf4
import re
model = ["AromeArctic", "MEPS"] #ECMWF later
source = ["thredds"] # later"netcdf", "grib" 2019120100
levtype = ["","pl","ml"] #include pl here aswell.


logging.basicConfig(filename="get_data.log", level = logging.INFO, format = '%(levelname)s : %(message)s')
#Nice error messeges.
def SomeError( exception = Exception, message = "Something did not go well" ):
    logging.error(exception(message))
    #source: https://softwareengineering.stackexchange.com/questions/222586/how-should-you-cleanly-restrict-object-property-types-and-values-in-python
    if isinstance( exception.args, tuple ):
        raise exception
    else:
        raise exception(message)
filter_function_for_type= lambda value: value if value in levtype else SomeError(ValueError, f'Type not found: choices:{levtype}')
filter_function_for_models = lambda value: value if value in model else SomeError(ValueError, f'Model not found: choices:{model}')
filter_function_for_source = lambda value: value if value in source else SomeError(ValueError, f'Source not found: choices:{source}')
filter_function_for_modelrun = lambda value: value \
    if ( len(value) == 10 ) and ( int(value[0:4]) in range(2000,2021) ) and ( int(value[4:6]) in range(0,13) ) \
    and ( int(value[6:8]) in range(1,32)) and ( int(value[9:10]) in range(0,25)) \
    else SomeError(ValueError, f'Modelrun wrong: Either; "latest or date on the form: YYYYMMDDHH')

def filter_for_bad_combination(data_domain, model, mbrs, levtype, source, modelrun, step,level, param):
    if source=="thredds" and model=="ooo": #on thredds modellevels are not available for members on MEPS
        check_available(modelrun)

        #SomeError(ValueError, f'Bad combination: On thredds modellevels is not available for members not being the deterministic(mbrs=0)')

    #if source=="thredds" and model=="MEPS" and mbrs != 0 and levtype== "ML": #on thredds modellevels are not available for members on MEPS
    #   SomeError(ValueError, f'Bad combination: On thredds modellevels is not available for members not being the deterministic(mbrs=0)')

class DATA():

    def __init__(self, data_domain, modelrun, param,  step, levtype="", level = [0], mbrs=0, model="AromeArctic", source="thredds", ):
        logging.info("START")
        self.data_domain = data_domain
        self.source = source
        self.model = model
        self.mbrs = mbrs
        self.modelrun = modelrun
        self.step = step
        self.levtype =  levtype
        self.level =  level
        self.param = param
        self.available_files = check_available(self.modelrun, self.mbrs,self.levtype, self.param, self.model)
        filter_for_bad_combination(self.data_domain, self.model, self.mbrs, self.levtype, self.source, self.modelrun, self.step, self.level, self.param)

    def __setattr__(self, key, value):
        #Here you can set the value of each parameter or alter the parameter from a userfriendly to pythonfriendly code.
        if key=="available_files":
            self.__dict__[key] = value
        if key=='levtype':
            value = filter_function_for_type(value)
            self.__dict__[key] = value
        if key=='model':
            value = filter_function_for_models(value)
            self.__dict__[key] = value
        if key=='source':
            value = filter_function_for_source(value)
            self.__dict__[key] = value
        if key=='modelrun':
            #if value !="latest":
            #    value = filter_function_for_modelrun(value)
            self.__dict__[key] = value
        if key=="data_domain": #or else obj would not be properly set...
            self.__dict__[key] = value
        if key == "param":  # or else obj would not be properly set...
            self.__dict__[key] = np.array(value)
        if key == "step":  # or else obj would not be properly set...
            self.__dict__[key] = value
        if key == "level":  # or else obj would not be properly set...
            self.__dict__[key] = value
        if key == "mbrs":
            self.__dict__[key] = value


    def make_url(self):
        '''
        Makes the OPENDAP url for the user specified model and parameters in a specific domain and time
        '''
        jindx = self.data_domain.idx[0]
        iindx = self.data_domain.idx[1]

        YYYY = self.modelrun[0:4]
        MM = self.modelrun[4:6]
        DD = self.modelrun[6:8]
        HH = self.modelrun[8:10]

        step = f"[{np.min(self.step)}:1:{np.max(self.step)}]"
        level = f"[{np.min(self.level)}:1:{np.max(self.level)}]"
        mbrs = f"[{np.min(self.mbrs)}:1:{np.max(self.mbrs)}]"
        y = f"[{jindx.min()}:1:{jindx.max()}]"
        x = f"[{iindx.min()}:1:{iindx.max()}]"
        non = f"[0:1:0]"

        ###############################################################################
        if self.model == "AromeArctic":
            file = self.available_files[0].copy()

            url = f"https://thredds.met.no/thredds/dodsC/aromearcticarchive/{YYYY}/{MM}/{DD}/{file.loc[0].at['File']}?"

            url += f"time{step}," + \
                f"latitude{y}{x}," + \
                f"longitude{y}{x}," +\
                f"x{x},"+ \
                f"y{y}," + \
                f"forecast_reference_time"

            if self.levtype=="ml":
                url += f",hybrid{level}," + \
                       f"ap{level}," + \
                       f"b{level}"
            startsub = ""
            for prm in self.param:
                url += f",{prm}"
                vardim = len(file.loc[0].at["var"][prm])
                if (vardim == 0):
                    startsub = f""
                if (vardim == 1):
                    startsub = f"{step}"
                if (vardim == 2):
                    startsub = f"{y}{x}"
                if (vardim == 3):
                    startsub = f"{step}{y}{x}"
                if (vardim == 4):
                    if (file.loc[0].at["mbr_bool"] == True):
                        startsub = f"{step}{mbrs}{y}{x}"
                    elif (re.match("^.*_ml|^.*_pl", prm)):
                        startsub = f"{step}{level}{y}{x}"
                    else:
                        startsub = f"{step}{non}{y}{x}"
                if (vardim == 5):
                    if (file.loc[0].at["mbr_bool"] == True):
                        startsub = f"{step}{mbrs}{non}{y}{x}"
                        if (re.match("^.*_ml|^.*_pl", prm)):
                            startsub = f"{step}{level}{mbrs}{y}{x}"
                    elif (re.match("^.*_ml|^.*_pl", prm)):
                        startsub = f"{step}{level}{non}{y}{x}"
                    else:
                        startsub = f"{step}{non}{non}{y}{x}"
                url += startsub
        ###############################################################################

        if self.model == "MEPS":
            #############FILTER###########################
            file = self.available_files[0].copy()
            #first filter
            logging.info(file)
            ####################################################################
            url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/{file.loc[0].at['File']}?"
            #for param in fixed_param:
            url += f"time{step}," + \
                    f"latitude{y}{x}," + \
                    f"longitude{y}{x}," + \
                    f"x{x}," + \
                    f"y{y}," + \
                    f"forecast_reference_time"

            if self.levtype=="ml":
                url += f",hybrid{level}," + \
                       f"ap{level}," + \
                       f"b{level}"

            startsub=""
            for prm in self.param:
                url += f",{prm}"
                vardim = len(file.loc[0].at["var"][prm])
                if (vardim == 0):
                    startsub = f""
                if (vardim == 1):
                    startsub = f"{step}"
                if (vardim == 2):
                    startsub = f"{y}{x}"
                if (vardim == 3):
                    startsub = f"{step}{y}{x}"
                if ( vardim == 4):
                    if (file.loc[0].at["mbr_bool"]==True):
                        startsub = f"{step}{mbrs}{y}{x}"
                    elif (re.match("^.*_ml|^.*_pl", prm)):
                        startsub = f"{step}{level}{y}{x}"
                    else:
                        startsub = f"{step}{non}{y}{x}"
                if (vardim == 5):
                    if (file.loc[0].at["mbr_bool"]==True):
                        startsub = f"{step}{mbrs}{non}{y}{x}"
                        if (re.match("^.*_ml|^.*_pl", prm)):
                            startsub = f"{step}{level}{mbrs}{y}{x}"
                    elif (re.match("^.*_ml|^.*_pl", prm)):
                        startsub = f"{step}{level}{non}{y}{x}"
                    else:
                        startsub = f"{step}{non}{non}{y}{x}"
                url+= startsub

        logging.info(url)
        self.__dict__["url"] = url
        return url

    def thredds(self, url):
        prm_fixed = ["time", "latitude", "longitude", "forecast_reference_time","x","y"]

        logging.info("-------> start retrieve from thredds")
        dataset = Dataset(url) #fast
        logging.info("-------> Getting variable: ")
        if self.model=="MEPS":
            prm_fixed = ["time", "latitude", "longitude", "forecast_reference_time", "x", "y"]
        iteration =-1
        for prm in prm_fixed:
            logging.info(prm)
            iteration +=1
            self.__dict__[prm] = dataset.variables[prm][:]

        if self.levtype=="ml":
            for prm in ["hybrid", "ap", "b" ]:
                logging.info(prm)
                self.__dict__[prm] = dataset.variables[prm][:]
            for prm in self.param:
                file = self.available_files[0].copy()

                print(file.loc[0].at["var"][prm])
                iteration += 1
                logging.info(prm)
                self.__dict__[prm] = dataset.variables[prm][:]

        dataset.close()
        iteration += 1


    def windcorr(self):
        jindx = self.data_domain.idx[0]
        iindx = self.data_domain.idx[1]
        if self.model == "AromeArctic":
            infile = "bin/alpha_full_AA.nc"
        elif self.model == "MEPS":
            infile = "bin/alpha_full_MEPS.nc"
        alphadata = Dataset(infile)
        alpha = alphadata["alpha"][:]
        self.__dict__["alpha"] = alpha[jindx.min():jindx.max()+1,iindx.min():iindx.max()+1]
        alphadata.close()

    def retrieve(self):
        if self.source == "thredds":
            url = self.make_url()
            self.thredds(url)
        self.windcorr()

