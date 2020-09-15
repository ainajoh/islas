"""
versions:
v1: AromeArcic on thredds on Svalbars.
v2: + MEPS at Finse with members
"""
from netCDF4 import Dataset
import numpy as np
import logging
import pandas as pd
import os
from imetkit.check_data import *  # require netcdf4
from imetkit.domain import *  # require netcdf4
import re

#mydir_new = os.chdir("/Users/ainajoh/Documents/GitHub/islas/scripts/FC-system")


model = ["AromeArctic", "MEPS"] #ECMWF later
source = ["thredds"] # later"netcdf", "grib" 2019120100
levtype = [None,"pl","ml"] #include pl here aswell.


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
filter_function_for_date = lambda value: value \
    if ( len(value) == 10 ) and ( int(value[0:4]) in range(2000,2021) ) and ( int(value[4:6]) in range(0,13) ) \
    and ( int(value[6:8]) in range(1,32)) and ( int(value[9:10]) in range(0,25)) \
    else SomeError(ValueError, f'Modelrun wrong: Either; "latest or date on the form: YYYYMMDDHH')

def filter_for_bad_combination(file, data_domain, model, mbrs, levtype, source, date, step,level, param):
    if source=="thredds" and model=="ooo": #on thredds modellevels are not available for members on MEPS
        check_available(date)
    #if leveltype = "pl" and level > :
        #SomeError(ValueError, f'Bad combination: On thredds modellevels is not available for members not being the deterministic(mbrs=0)')

    #if source=="thredds" and model=="MEPS" and mbrs != 0 and levtype== "ML": #on thredds modellevels are not available for members on MEPS
    #   SomeError(ValueError, f'Bad combination: On thredds modellevels is not available for members not being the deterministic(mbrs=0)')


class get_data():

    def __init__(self, model, date, param, file, step, data_domain=None, levtype=None, level = None, mbrs=0, source="thredds" ):
        logging.info("START")
        self.source = source
        self.model = model
        self.mbrs = mbrs
        self.date = date
        self.step = step
        self.levtype =  levtype
        self.level =  level
        self.param = param
        self.file = file

        url = "https://thredds.met.no/thredds/catalog/meps25epsarchive/catalog.html"
        webcheck = requests.head(url)
        if webcheck.status_code != 200:  # SomeError(ValueError, f'Type not found: choices:{levtype}')
            SomeError(ConnectionError, f"Website {url} is down with {webcheck}; . Wait until it is up again")


        if data_domain:
            #self.data_domain = data_domain
            self.idx = data_domain.idx
            self.lonlat = data_domain.lonlat
        else:
            xmax = file["var"][0]["x"]["shape"][0]-1
            ymax=  file["var"][0]["y"]["shape"][0]-1
            self.idx = ( np.array([0,ymax]), np.array([0,xmax]) )
            self.lonlat = None

        if levtype =="pl" and self.level == None:
            maxlvl = file["var"][0]["pressure"]["shape"][0] - 1
            self.level = [0,maxlvl]
        if levtype == "ml" and self.level == None:
            maxlvl = file["var"]["hybrid"]["shape"][0] -1
            self.level = [0,maxlvl]
        if self.source == "thredds":
            self.url = self.make_url()
        filter_for_bad_combination(self.file, data_domain, self.model, self.mbrs, self.levtype, self.source, self.date, self.step, self.level, self.param)

    def __setattr__(self, key, value):
        #Here you can set the value of each parameter or alter the parameter from a userfriendly to pythonfriendly code.
        if key=="idx":
            self.__dict__[key] = value
        if key=="lonlat":
            self.__dict__[key] = value
        if key=="file":
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
        if key=='date':
            self.__dict__[key] = value
        if key == "param":  # or else obj would not be properly set...
            self.__dict__[key] = np.array(value)
        if key == "step":  # or else obj would not be properly set...
            self.__dict__[key] = value
        if key == "level":  # or else obj would not be properly set...
            #if value == None  ype(dii["pressure"][0])
            value = 0
            self.__dict__[key] = value
        if key == "mbrs":
            if value == None:
                value = 0
            self.__dict__[key] = value
    def handle_noneinputs(self):
        pass
    def make_url(self):
        '''
        Makes the OPENDAP url for the user specified model and parameters in a specific domain and time
        '''

        jindx = self.idx[0]
        iindx = self.idx[1]

        YYYY = self.date[0:4]
        MM = self.date[4:6]
        DD = self.date[6:8]
        HH = self.date[8:10]

        step = f"[{np.min(self.step)}:1:{np.max(self.step)}]"
        level = f"[{np.min(self.level)}:1:{np.max(self.level)}]"
        mbrs = f"[{np.min(self.mbrs)}:1:{np.max(self.mbrs)}]"
        y = f"[{jindx.min()}:1:{jindx.max()}]"
        x = f"[{iindx.min()}:1:{iindx.max()}]"
        non = f"[0:1:0]"
        indexidct = {"time": step, "y": y, "x": x, "ensemble_member": mbrs,
                     "pressure": level, "hybrid": level, "hybrid0": non,
                     "height0": non, "height1": non, "height2": non,
                     "height3": non, "height7": non, 'height_above_msl': non}

        fixed_var = np.array(["latitude","longitude","forecast_reference_time","projection_lambert", "ap","b"]) #if excicst use.
        fixed_var = fixed_var[np.isin(fixed_var,list(self.file["var"][0].keys()))]
        self.param = np.append(self.param, fixed_var)
        #print(varinfile)
        ###############################################################################
        if self.model == "AromeArctic":
            file = self.file.copy()
            param = self.param.copy()
            url = f"https://thredds.met.no/thredds/dodsC/aromearcticarchive/{YYYY}/{MM}/{DD}/{file.loc[0].at['File']}?"
            startsub = ""
            for prm in param:
                #print(prm)
                url += f"{prm}"
                dimlist = list(file["var"][0][prm]["dim"])  # ('time', 'pressure', 'ensemble_member', 'y', 'x')
                # print(file["dim"][0])
                newlist = [indexidct[i] for i in dimlist]
                startsub = ''.join(newlist) + ","
                #np.setdiff1d(list_2, list_1)
                for dimen in np.setdiff1d(file["var"][0][prm]["dim"], self.param):
                    #self.__setattr__(self, param, value)
                    self.param = np.append(self.param, dimen)
                    startsub += dimen
                    startsub += indexidct[dimen]+ ","

                url += startsub

        ###############################################################################

        if self.model == "MEPS":
            #############FILTER###########################
            file = self.file.copy()
            #print()
            #first filter
            logging.info(file)
            ####################################################################
            url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/{file.loc[0].at['File']}?"
            #for param in fixed_param:

            startsub=""
            for prm in self.param:
                url += f"{prm}"
                dimlist = list(file["var"][0][prm]["dim"])  #('time', 'pressure', 'ensemble_member', 'y', 'x')
                #print(file["dim"][0])
                newlist = [indexidct[i] for i in dimlist]
                startsub = ''.join(newlist)
                for dimen in np.setdiff1d(file["var"][0][prm]["dim"], self.param):
                    self.param = np.append(self.param, dimen)
                    startsub += dimen
                    startsub += indexidct[dimen] + ","

                url+= startsub

        url = url.rstrip(",")
        logging.info(url)
        self.__dict__["url"] = url

        return url#.rstrip(',')

    def thredds(self, url):

        logging.info("-------> start retrieve from thredds")
        dataset = Dataset(url) #fast
        for k in dataset.__dict__.keys():
            ss = f"{k}"
            self.__dict__[ss] = dataset.__dict__[k]
        logging.info("-------> Getting variable: ")
        iteration =-1

        for prm in self.param:
            iteration += 1
            logging.info(prm)
            for k in dataset.variables[prm].__dict__.keys():
                ss = f"{k}_{prm}"
                self.__dict__[ss] = dataset.variables[prm].__dict__[k]
            #self.__dict__[f"unit_{prm}"] = dataset.variables[prm].units
            self.__dict__[prm] = dataset.variables[prm][:]

        dataset.close()
        iteration += 1


    def windcorr(self):
        jindx = self.idx[0]
        iindx = self.idx[1]
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
            #self.url = self.make_url()
            self.thredds(self.url)
        self.windcorr()

    def info(self):
        prm_fixed = ["time", "latitude", "longitude", "forecast_reference_time", "x", "y", "projection_lambert"]

        logging.info("-------> start retrieve from thredds")
        dataset = Dataset(self.url)  # fast
        self.info = dataset
        logging.info("-------> Getting variable: ")
        if self.model == "MEPS":
            prm_fixed = ["time", "latitude", "longitude", "forecast_reference_time", "x", "y"]
        iteration = -1
        for prm in prm_fixed:
            logging.info(prm)
            iteration += 1
            self.__dict__[prm] = dataset.variables[prm]

        if self.levtype == "ml":
            for prm in ["hybrid", "ap", "b"]:
                logging.info(prm)
                self.__dict__[prm] = dataset.variables[prm]
        for prm in self.param:
            iteration += 1
            logging.info(prm)
            self.__dict__[prm] = dataset.variables[prm]

        #self.info = dataset
        dataset.close()
        iteration += 1



