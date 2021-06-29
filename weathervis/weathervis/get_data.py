########################################################################
# File name: get_data.py
# This file is part of: weathervis
########################################################################
from netCDF4 import Dataset
import numpy as np
import logging
import pandas as pd
import os
from weathervis.check_data import *  # require netcdf4
from weathervis.domain import *  # require netcdf4
import re
import pkgutil

"""
###################################################################
This module gets the data defined by the user 
------------------------------------------------------------------------------
Usage:
------
data =  get_data(model, date, param, file, step, data_domain=None, p_level = None, m_level = None, mbrs=None)
Returns:
------
data Object with properties
"""

package_path = os.path.dirname(__file__)
# Nice logging info saved to aditional file
logging.basicConfig(filename="get_data.log", level = logging.INFO, format = '%(levelname)s : %(message)s')

#use_latest = True

def SomeError( exception = Exception, message = "Something did not go well" ):
    # Nice error messeges.
    logging.error(exception(message))
    #source: https://softwareengineering.stackexchange.com/questions/222586/how-should-you-cleanly-restrict-object-property-types-and-values-in-python
    if isinstance( exception.args, tuple ):
        raise exception
    else:
        raise exception(message)

# Valid options of certain variables
available_models = ["AromeArctic", "MEPS"] #ECMWF later
# functions for filtering out unvalid uptions
check_if_thredds_is_down = lambda value:value if requests.head(value) != 200 else SomeError(ConnectionError, f"Website {value} is down;. Wait until it is up again")
filter_function_for_type= lambda value: value if value in levtype else SomeError(ValueError, f'Type not found: choices:{levtype}')
filter_function_for_models = lambda value: value if value in available_models else SomeError(ValueError, f'Model not found: choices:{model}')
filter_function_for_date = lambda value: value \
    if ( len(value) == 10 ) and ( int(value[0:4]) in range(2000,2030) ) and ( int(value[4:6]) in range(0,13) ) \
    and ( int(value[6:8]) in range(1,32)) and ( int(value[9:10]) in range(0,25)) \
    else SomeError(ValueError, f'Modelrun wrong: Either; "latest or date on the form: YYYYMMDDHH')
filter_function_for_mbrs=lambda value, file: value if max(value) < file.dim["ensemble_member"]["shape"] else SomeError(ValueError, f'Member input outside range of model')
filter_function_for_step=lambda value, file: value if np.max(value) < file.dim["time"]["shape"] else SomeError(ValueError, f' step input outside range of model')
filter_function_for_p_level=lambda value, file: value if set(value).issubset(set(file["p_levels"])) else SomeError(ValueError, f' p_level input outside range of model')
filter_function_for_m_level=lambda value, file: value if np.max(value) < file.dim["hybrid"]["shape"] else SomeError(ValueError, f' m_level input outside range of model')
filter_function_for_param=lambda value, file: value if set(value).issubset(set(file["var"].keys())) else SomeError(ValueError, f' param input not possible for this file')
filter_function_for_file=lambda value: value if value is not None else SomeError(ValueError, f' File has to be given (see check_data(()')

#filter_function_for_domain=lambda value: value if value in np.array(dir(domain))  else SomeError(ValueError, f'Domain name not found')
#Domain filter not needed as it should be handled in domain itself
class get_data():

    def __init__(self, model=None, date=None, param=None, file=None, step=None, data_domain=None, p_level = None, m_level = None, mbrs=None, url=None, use_latest=True):
        """
        Parameters - Type - Info - Example
        ----------
        model: - String - The weather model we want data from            - Example: model = "MEPS"
        date:  - String - date and time of modelrun in format YYYYMMDDHH - Example: date = "2020012000"
        param: - List of Strings - Parameters we want from model         - Example: param = ["wind_10m"]
        file:  - Panda Series of strings and dictionaries -  Returned from chech_data.py -
        step   - int or list of ints - Forecast time steps -
        data_domain - String - Domain name as defined in the domain.py file
        p_level - int or list of ints - Pressure levels
        m_level -  int or list of ints - model levels
        mbrs     - int or list of ints - ensemble members
        url      - url of where we can find file on thredds
        """

        logging.info("START")
        # Initialising -- NB! The order matters ###
        self.model = model
        self.mbrs = mbrs
        self.date = date
        self.step = step
        self.p_level = p_level if type(p_level)==list or p_level==None else list(p_level)
        self.m_level = m_level
        self.param = np.unique(param).tolist()
        self.data_domain = data_domain
        #If file comes in as a dataframe with possibly multiple rows, choose one and make it a Series
        self.file = file.loc[0] if type(file) == pd.core.frame.DataFrame else file
        #If no datadomain is set, choose idx to span the entire modeldomain
        self.idx = data_domain.idx if data_domain else ( np.array([0, self.file["var"]["y"]["shape"][0]-1]), np.array([0,self.file["var"]["x"]["shape"][0]-1]) )
        self.lonlat = data_domain.lonlat if data_domain else None
        #if no member is wanted initially, then we exclude an aditional dimension caused by this with mbrs_bool later
        self.mbrs_bool = False if self.mbrs == None else True
        #If No member is wanted, we take the control (mbr=0)
        self.mbrs = 0 if self.mbrs == None else mbrs
        self.url = url
        self.units = self.dummyobject()
        self.FillValue = self.dummyobject()
        self.use_latest=use_latest


        #Check and filter for valid settings. If any of these result in a error, this script stops
        check_if_thredds_is_down("https://thredds.met.no/thredds/catalog/meps25epsarchive/catalog.html")

        if self.url is None:
            filter_function_for_file(self.file)
            filter_function_for_models(self.model)
            filter_function_for_mbrs(np.array(self.mbrs), self.file) if self.mbrs != None and self.mbrs_bool else None
            filter_function_for_date(self.date)
            filter_function_for_step(self.step,self.file)

            filter_function_for_p_level(np.array(self.p_level),self.file) if self.p_level != None and self.file.p_levels != None else None
            filter_function_for_m_level(np.array(self.m_level),self.file) if self.m_level != None and self.file.ml_bool else None

            filter_function_for_param(self.param, self.file) #this is kind of checked in check_data.py already.

            #Adjusting some parameters.
            #This is an option since users might not now how many pressure/model levels the model has and just want all.
            if self.p_level == None or self.file.p_levels == None :
                # If no pressure level is defined initially, we get all pressure levels for the paramter that depends on pressure.
                if "pressure" in self.file["dim"].keys() :
                    self.p_level = self.file["p_levels"]
                else: #if no parameter depends on pressure p_level = 0
                    self.p_level = None#[0]

            if self.m_level == None:
                # If no model level is defined initially, we get all model levels for the paramter that depends on modellevels.
                if "hybrid" in self.file["dim"].keys() :
                    maxpl = self.file["dim"]["hybrid"]["shape"] -1
                    self.m_level = [0,maxpl]
                else: #if no parameter depends on modellevel set it to 0
                    maxpl = 0
                    self.m_level = [0]
                if "hybrid2" in self.file["dim"].keys() and maxpl <=1:
                    maxpl = self.file["dim"]["hybrid2"]["shape"] -1
                    self.m_level = [0,maxpl]


        #Make a url depending on preferences if no url is defined already.
        self.url = self.make_url() if self.url == None else self.adjust_user_url()

    def adjust_user_url(self):
        if self.param is None and self.steps is None and self.data_domain is None and self.p_level is None and self.m_level is None and self.mbrs is None:
            return self.url

        jindx = self.idx[0]
        iindx = self.idx[1]

        if self.p_level:
            idx = np.where( np.array(self.file["p_levels"])[:, None] == np.array(self.p_level)[None, :])[0]
        else:
            idx=0

        # Sets up the userdefined range of value in thredds format [start:step:stop]
        step = f"[{np.min(self.step)}:1:{np.max(self.step)}]"
        pl_idx = f"[{np.min(idx)}:1:{np.max(idx)}]"
        m_level = f"[{np.min(self.m_level)}:1:{np.max(self.m_level)}]"
        mbrs = f"[{np.min(self.mbrs)}:1:{np.max(self.mbrs)}]"
        y = f"[{jindx.min()}:1:{jindx.max()}]"
        x = f"[{iindx.min()}:1:{iindx.max()}]"
        non = f"[0:1:0]"

        # indexidct Keeps track of dimensions for the different independent variables.
        indexidct = {"time": step, "y": y, "x": x, "ensemble_member": mbrs,
                     "pressure": pl_idx, "hybrid": m_level, "hybrid2": m_level, "hybrid0": non,
                     "height0": non, "height1": non, "height2": non,
                     "height3": non, "height7": non, "height6": non, 'height_above_msl': non, "mean_sea_level": non,
                     "atmosphere_as_single_layer": non,"top_of_atmosphere":non}
        # fixed_var: The fixed variables we always want
        fixed_var = np.array(
            ["latitude", "longitude", "forecast_reference_time", "projection_lambert", "ap", "b", "ap2", "b2"])
        # keep only the fixed variables that are actually available in the file.
        fixed_var = fixed_var[np.isin(fixed_var, list(self.file["var"].keys()))]
        # update global variable to include fixed var
        self.param = np.append(self.param, fixed_var)  # Contains absolutely all variables we want

        file = self.file.copy()
        param = self.param.copy()
        logging.info(file)
        url = f"{self.url}?"
        for prm in param:  # loop that updates the url to include each parameter with its dimensions
            url += f"{prm}"  # example:  url =url+x_wind_pl
            dimlist = list(file["var"][prm][
                               "dim"])  # List of the variables the param depends on ('time', 'pressure', 'ensemble_member', 'y', 'x')
            newlist = [indexidct[i] for i in
                       dimlist]  # convert dependent variable name to our set values. E.g: time = step = [0:1:0]
            startsub = ''.join(
                newlist) + ","  # example: ('time', 'pressure','ensemble_member','y','x') = [0:1:0][0:1:1][0:1:10][0:1:798][0:1:978]
            for dimen in np.setdiff1d(file["var"][prm]["dim"], self.param):
                # includes the dim parameters like, pressure, hybrid, height as long as we havent already gone through them
                self.param = np.append(self.param,
                                       dimen)  # update global param with the var name so that we do not go through it multiple time.
                startsub += dimen
                startsub += indexidct[dimen] + ","
            url += startsub

        url = url.rstrip(",")  # if url ends with , it creates error so remove.
        logging.info(url)
        # self.__dict__["url"] = url
        return url  # returns the url that will be set to global url.



    def make_url(self):
        """
        Makes the OPENDAP url for the user specified model and parameters in a specific domain and time
        Returns
        -------
        url for thredds
        """
        #Initialising variables
        jindx = self.idx[0]
        iindx = self.idx[1]

        YYYY = self.date[0:4]
        MM = self.date[4:6]
        DD = self.date[6:8]
        HH = self.date[8:10]

        if self.p_level:
            idx = np.where( np.array(self.file["p_levels"])[:, None] == np.array(self.p_level)[None, :])[0]
        else:
            idx=0
        #Sets up the userdefined range of value in thredds format [start:step:stop]
        step = f"[{np.min(self.step)}:1:{np.max(self.step)}]"
        pl_idx = f"[{np.min(idx)}:1:{np.max(idx)}]"
        m_level  = f"[{np.min(self.m_level)}:1:{np.max(self.m_level)}]"

        mbrs = f"[{np.min(self.mbrs)}:1:{np.max(self.mbrs)}]"
        y = f"[{jindx.min()}:1:{jindx.max()}]"
        x = f"[{iindx.min()}:1:{iindx.max()}]"
        non = f"[0:1:0]"
        #indexidct Keeps track of dimensions for the different independent variables.
        indexidct = {"time": step, "y": y, "x": x, "ensemble_member": mbrs,
                     "pressure": pl_idx, "hybrid": m_level, "hybrid2": m_level,"hybrid0": non,
                     "height0": non, "height1": non, "height2": non,
                     "height3": non, "height7": non, "height6": non,'height_above_msl': non, "mean_sea_level":non,
                     "atmosphere_as_single_layer":non,"top_of_atmosphere":non}

        # fixed_var: The fixed variables we always want
        fixed_var = np.array(["latitude","longitude","forecast_reference_time","projection_lambert", "ap","b", "ap2","b2"])
        # keep only the fixed variables that are actually available in the file.
        fixed_var = fixed_var[np.isin(fixed_var, list(self.file["var"].keys()))]
        # update global variable to include fixed var
        self.param = np.append(self.param, fixed_var) #Contains absolutely all variables we want.
        self.param = np.unique(self.param).tolist()

        ###############################################################################
        if self.model == "AromeArctic":
            file = self.file.copy()
            param = self.param.copy()
            logging.info(file)

            if self.use_latest==False:
              url = f"https://thredds.met.no/thredds/dodsC/aromearcticarchive/{YYYY}/{MM}/{DD}/{file.loc['File']}?"
            else:
              url = f"https://thredds.met.no/thredds/dodsC/aromearcticlatest/{file.loc['File']}?"

            for prm in param: #loop that updates the url to include each parameter with its dimensions
                url += f"{prm}"                           # example:  url =url+x_wind_pl
                dimlist = list(file["var"][prm]["dim"])   # List of the variables the param depends on ('time', 'pressure', 'ensemble_member', 'y', 'x')
                newlist = [indexidct[i] for i in dimlist] # convert dependent variable name to our set values. E.g: time = step = [0:1:0]
                startsub = ''.join(newlist) + ","         # example: ('time', 'pressure','ensemble_member','y','x') = [0:1:0][0:1:1][0:1:10][0:1:798][0:1:978]
                for dimen in np.setdiff1d(file["var"][prm]["dim"], self.param):
                    # includes the dim parameters like, pressure, hybrid, height as long as we havent already gone through them
                    self.param = np.append(self.param, dimen) #update global param with the var name so that we do not go through it multiple time.
                    startsub += dimen
                    startsub += indexidct[dimen]+ ","
                url += startsub

        if self.model == "MEPS":  #This has become completely equal to AromeArctic except for base url. Consider adding them in the future.
            file = self.file.copy()
            param = self.param.copy() #has to be copied in order for not infinite loop when updating self.param
            logging.info(file)

            if self.use_latest==False:
              url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/{file.loc['File']}?"
            else:
              url = f"https://thredds.met.no/thredds/dodsC/meps25epslatest/{file.loc['File']}?"

            for prm in param:  #loop that updates the url to include each parameter with its dimensions
                url += f"{prm}"                           # example:  url =url+x_wind_pl
                dimlist = list(file["var"][prm]["dim"])   # List of the variables the param depends on ('time', 'pressure', 'ensemble_member', 'y', 'x')
                newlist = [indexidct[i] for i in dimlist] # convert dependent variable name to our set values. E.g: time = step = [0:1:0]
                startsub = ''.join(newlist) + ","         # example: ('time', 'pressure','ensemble_member','y','x') = x_wind_pl[0:1:0][0:1:1][0:1:10][0:1:798][0:1:978]
                for dimen in np.setdiff1d(file["var"][prm]["dim"], self.param):
                    # includes the dim parameters like, pressure, hybrid, height as long as we havent already gone through them
                    self.param = np.append(self.param, dimen) #update global param with the var name so that we do not go through it multiple time.
                    startsub += dimen + indexidct[dimen]+ ","
                url += startsub #add parameters to main url.

        url = url.rstrip(",") #if url ends with , it creates error so remove.
        logging.info(url)
        #self.__dict__["url"] = url
        return url #returns the url that will be set to global url.

    def thredds(self, url, file):
        """
        Retrieves the data from thredds and set it as attributes to the global object.
        Parameters
        ----------
        url:
        file
        Returns
        -------
        """

        logging.info("-------> start retrieve from thredds")
        dataset = Dataset(url) #fast
        for k in dataset.__dict__.keys(): #info of the file
            ss = f"{k}"
            self.__dict__[ss] = dataset.__dict__[k]
        logging.info("-------> Getting variable: ")
        iteration =-1
        for prm in self.param:
            iteration += 1
            logging.info(prm)
            if "units" in dataset.variables[prm].__dict__.keys():
                self.units.__dict__[prm] = dataset.variables[prm].__dict__["units"]
            if "_FillValue" in dataset.variables[prm].__dict__.keys():
                self.FillValue.__dict__[prm] = int(dataset.variables[prm].__dict__["_FillValue"])
            else:
                self.FillValue.__dict__[prm] = np.nan


            if prm == "projection_lambert":
                for k in dataset.variables[prm].__dict__.keys():
                    ss = f"{k}_{prm}"
                    self.__dict__[ss] = dataset.variables[prm].__dict__[k]
            varvar = dataset.variables[prm][:] ##
            dimlist = np.array(list(file["var"][prm]["dim"]))  # ('time', 'pressure', 'ensemble_member', 'y', 'x')
            if not self.mbrs_bool and any(np.isin(dimlist, "ensemble_member")):#"ensemble_member" in dimlist:
                indxmember = np.where(dimlist == "ensemble_member")[0][0]
                varvar = dataset.variables[prm][:].squeeze(axis=indxmember)


            self.__dict__[prm] = varvar

        dataset.close()
        iteration += 1


    def windcorr(self):
        jindx = self.idx[0]
        iindx = self.idx[1]
        if self.model == "AromeArctic":
            infile = package_path + "/data/alpha_full_AA.nc"
        elif self.model == "MEPS":
            infile = package_path + "/data/alpha_full_MEPS.nc"
        alphadata = Dataset(infile)
        alpha = alphadata["alpha"][:]
        self.__dict__["alpha"] = alpha[jindx.min():jindx.max()+1,iindx.min():iindx.max()+1]

        alphadata.close()

    def retrieve(self):
        #self.url = self.make_url()
        self.thredds(self.url, self.file)
        self.windcorr()

    class dummyobject(object):pass