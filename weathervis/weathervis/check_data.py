########################################################################
# File name: check_data.py
# This file is part of: imetkit
########################################################################
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import logging
from collections import Counter
import os
"""
###################################################################
This module checks what data is available, gives information on the dataset 
and choose the user prefered dataset.
------------------------------------------------------------------------------
Usage:
------
check = check_data(model, date = None, param=None, mbrs=None,levtype=None, file = None, numbervar = 100, search = None)

Returns:
------
Object with properties
"""
package_path = os.path.dirname(__file__)


model = ["AromeArctic", "MEPS"] #ECMWF later
source = ["thredds"] # later"netcdf", "grib" 2019120100
levtype = [None,"pl","ml"] #include pl here aswell.

logging.basicConfig(filename="get_data.log", level = logging.INFO, format = '%(levelname)s : %(message)s')

def SomeError( exception = Exception, message = "Something did not go well" ):
    #source: https://softwareengineering.stackexchange.com/questions/222586/how-should-you-cleanly-restrict-object-property-types-and-values-in-python
    logging.error(exception(message))
    if isinstance( exception.args, tuple ):
        raise exception
    else:
        raise exception(message)

def filter_param(file,param):
    """Used by check_data: Remove files not containing a givet set of parameters.
    Returns only files containing all the user defined parameters."""
    if param:  #If a user param is given
        for i in range(0, len(file)): # go through all files,
            param_bool = np.array([key in file.loc[i,"var"].keys() for key in param])
            if all(param_bool) == False:  #remove those files not having that parameter.
                file.drop([i], inplace=True)
    file.reset_index(inplace=True, drop=True)
    logging.info(file)
    return file

def filter_type(file,mbrs,levtype, p_level,m_level):
    """Used by check_data: Remove files not having the userdefined mbrs and levtype
    Returns only files containing all the user defined preferences."""
    if mbrs != 0 and mbrs != None:
        file = file[file["mbr_bool"] == True]
    if m_level != None:
        file = file[file["ml_bool"] == True]
    elif p_level:
        file = file[~file.p_levels.isnull()]
        file.reset_index(inplace=True)
        ll = file.p_levels.tolist()
        file = file[pd.DataFrame(ll).isin(p_level).sum(axis=1)==len(p_level)]
    file.reset_index(inplace=True, drop=True)
    return file

def filter_step(file,maxstep):
    if maxstep != None:
        for i in range(0, len(file)): # go through all files,
            step_bool = int(file.loc[i,"dim"]["time"]["shape"]) >= maxstep
            if step_bool == False:
                file.drop([i], inplace=True)
    file.reset_index(inplace=True, drop=True)
    return file

filter_function_for_models = lambda value: value if value in model else SomeError(ValueError, f'Model not found: choices:{model}')
filter_function_for_models = lambda value: value if value in model else SomeError(ValueError, f'Model not found: choices:{model}')


def filter_any(file):
    """Used by check_data: Remove random files until only one left
    Returns only one file.
    Todo: find a better way as this might not be what the use expect"""
    if len(file) > 1:  # want to end up with only one file.
        file = file[0]
        file.reset_index(inplace=True, drop=True)
    return file


class check_data():

    def __init__(self, model, date = None,param=None, step=None, mbrs=None,levtype=None, p_level= None, m_level=None,file = None, numbervar = 100, search = None):
        """
        Parameters
        ----------
        model: Weathermodel, either: MEPS, AromeArctic
        date:  Modelrun as string in format YYYYMMDDHH
        param: Parameters as strings in a list
        mbrs:  Which ensemble member
        levtype: What type of vertical level
        file:    If you already know the filename you want
        numbervar: max number of listed pameter, for searching.
        search: Parameter to search for.
        """
        self.date = date
        self.model = model
        self.param = param
        self.mbrs = mbrs
        self.levtype = levtype
        self.file = file
        self.numbervar = numbervar
        self.search = search
        self.p_level = p_level
        self.m_level = m_level
        self.maxstep = np.max(step) if step != None or type(step) != float or type(step) != int else step


        if p_level:
            self.levtype="pl"
            if type(p_level) != list:
                    self.p_level = [p_level]
        if m_level:
            self.levtype = "ml"
            if type(p_level) != list:
                self.m_level = [m_level]


        url = "https://thredds.met.no/thredds/catalog/meps25epsarchive/catalog.html"
        try:
            webcheck = requests.head(url,timeout=5)
        except requests.exceptions.Timeout as e:
            print(e)
            print("might be problems with the server; check out https://status.met.no")

        if webcheck.status_code != 200:  # SomeError(ValueError, f'Type not found: choices:{levtype}')
            SomeError(ConnectionError, f"Website {url} is down with {webcheck}; . Wait until it is up again. Check https://status.met.no")


        if date != None:
            print("CHECK_FILE STAAAART")
            self.file = self.check_files(date, model, param,  mbrs,levtype)
        if self.param == None and self.date != None:
            self.param = self.check_variable(self.file, self.search)

        if self.date == None and self.param ==None:
            self.param = self.check_variable_all(self.model, self.numbervar, self.search)
            if self.search:
                self.date = self.check_available_date(self.model, self.search)
            else:
                self.date = self.check_available_date(self.model)


    def check_available_date(self, model, search = None):
        df = pd.read_csv(f"{package_path}/data/{model}_filesandvar.csv")

        dfc = df.copy()  # df['base_name'] = [re.sub(r'_[0-9]*T[0-9]*Z.nc','', str(x)) for x in df['File']]
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk","_preop_"]
        df_base = pd.DataFrame([re.sub(r'_[0-9]*T[0-9]*Z.nc', '', str(x)) for x in df['File']], columns=["base_name"])
        dfc["base_name"] = df_base["base_name"]

        dfc = dfc[~dfc["base_name"].str.contains('|'.join(drop_files))]  # (drop_files)])
        if search:
            dfc = dfc[dfc["var"].str.contains(search)==True]
        dfc.reset_index(inplace=True, drop=True)


        #print(dfc)
        #df_base = dfc['var'].str.replace(" ", "").str.split(",")  # , expand = True)
        dateti = dfc[["Date","Hour"]].copy()
        dateti.drop_duplicates(keep='first', inplace=True)
        dateti.reset_index(inplace=True, drop=True)

        return dateti

    def check_filecontainingvar(self, model, numbervar, search ):
        #NOT IN USE
        #Nice to have a list of the files containing that var, but that means scraping the web too often.
        #Maybe add on file. scraping only new dates...DID It! Just need to update this function to find file containing: Then another function saying at what date.

        # Todo: update R scripts to only add new lines in filevar info
        df = pd.read_csv(f"bin/{model}_filesandvar.csv")
        dfc = df.copy()
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk","_preop_"]
        df_base = pd.DataFrame([re.sub(r'_[0-9]*T[0-9]*Z.nc', '', str(x)) for x in df['File']], columns=["base_name"])
        dfc["base_name"] = df_base["base_name"]
        dfc = dfc[~dfc["base_name"].str.contains('|'.join(drop_files))]  # (drop_files)])

        df_base = dfc['var'].str.replace(" ", "").str.split(",")  # , expand = True)
        search = "wind"
        #param = df_base[df_base.apply(lambda x: x.str.contains(search)).any(axis=1)]
        test = ["heipadeg", "du", "hei du"]

        flattened = [val for sublist in df_base[:] for val in sublist]



    def check_variable_all(self, model, numbervar, search ):
        df = pd.read_csv(f"{package_path}/data/{model}_filesandvar.csv")
        dfc = df.copy()  # df['base_name'] = [re.sub(r'_[0-9]*T[0-9]*Z.nc','', str(x)) for x in df['File']]
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk","_preop_"]
        df_base = pd.DataFrame([re.sub(r'_[0-9]*T[0-9]*Z.nc', '', str(x)) for x in df['File']], columns=["base_name"])
        dfc["base_name"] = df_base["base_name"]
        dfc = dfc[~dfc["base_name"].str.contains('|'.join(drop_files))]  # (drop_files)])
        df_base = dfc['var'].str.replace(" ", "").str.split(",")  # , expand = True)
        flattened = [val for sublist in df_base[:] for val in sublist]
        if search:
            flattened = [s for s in flattened if str(search) in s]
            count = Counter(flattened).most_common(len(flattened))
            param = pd.DataFrame(count, columns = ["param", "used"])["param"]
        else:
            count = Counter(flattened).most_common(numbervar)
            param = pd.DataFrame(count, columns = ["param", "used"])["param"]

        return param.to_string()


    def check_variable(self, file, search):
        var_dict = file.at[0, "var"]
        param = []
        for n in range(0,len(file)):
            filename =  file.at[n, "File"]
            var = file.at[n, "var"].keys()
            param.append( pd.DataFrame([x for x in var], columns=[filename]))

        param = pd.concat(param, axis = 1, join = "outer", sort=True)
        if search:
            param = param[param.apply(lambda x: x.str.contains(search))]#.any(axis=1)]
            param = param.dropna(how='all')
            #param = param[param.str.contains("wind")]
            #df1[df1['col'].str.contains(r'foo(?!$)')]

        return param.to_string()

    def check_files(self, date, model, param, mbrs,levtype):
        YYYY = date[0:4]
        MM = date[4:6]
        DD = date[6:8]
        HH = date[8:10]
        base_url=""
        if model=="MEPS":
            base_url = "https://thredds.met.no/thredds/catalog/meps25epsarchive/"   #info about date, years and filname of our model
            base_urlfile = "https://thredds.met.no/thredds/dodsC/meps25epsarchive/" #info about variables in each file
        elif model == "AromeArctic":
            base_url = "https://thredds.met.no/thredds/catalog/aromearcticarchive/"
            base_urlfile = "https://thredds.met.no/thredds/dodsC/aromearcticarchive/"
        else:
            pass
        print(base_url)
        print("base_url SHOULD BE PRRRINTED")


        #Find what files exist at that date
        page = requests.get(base_url + YYYY+"/"+MM+"/"+DD+ "/catalog.html")
        soup = BeautifulSoup(page.text, 'html.parser')
        rawfiles= soup.table.find_all("a")
        ff =[i.text for i in rawfiles]
        pattern=re.compile(f'.*{YYYY}{MM}{DD}T{HH}Z.nc')
        ff= pd.DataFrame( data = list(filter(pattern.match,ff)), columns=["File"])
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk","_preop_"]
        df = ff.copy()[~ff["File"].str.contains('|'.join(drop_files))] #(drop_files)])

        df.reset_index(inplace=True, drop = True)
        df["var"] = None
        df["dim"] = None
        df["mbr_bool"] = False
        df["ml_bool"] = False
        df["p_levels"] = False
        i=0
        while i<len(df):
            file=df["File"][i]
            url = base_urlfile + YYYY+"/"+MM+"/"+DD+ "/"+ file
            dataset = Dataset(url)
            #for all independent var (dimensions) make a column with dict
            dn = dataset.dimensions.keys()
            ds = [dataset.dimensions[d].size for d in dn]
            valued = np.full(np.shape(ds), np.nan)
            dimdic = dict(zip(dn,ds))
            dimlist =  list(zip(ds, valued))

            dimframe = pd.DataFrame(dimlist,index = dn, columns=["shape","value"])
            #dimframe = pd.DataFrame(ds, index = dn,columns=["shape"])
            #check leveltype
            df.loc[i,"mbr_bool"] = ("ensemble_member" in dimdic) and (dimdic["ensemble_member"]>=10)
            df.loc[i,"ml_bool"] = ("hybrid" in dimdic) and (dimdic["hybrid"]>=65)
            df['p_levels'] = df['p_levels'].astype(object)
            df.at[i,"p_levels"] =  [int(x) for x in dataset.variables["pressure"][:]] if "pressure" in dimdic and dimdic["pressure"]>=1 else None
            #df.loc[i,"p_levels"] = ("pressure" in dimdic) and (dimdic["pressure"]>=10)
            av_pl_levels = dataset.variables["pressure"][:] if "pressure" in dimdic else None
            if "pressure" in dimdic:
                dimframe.loc["pressure","value"] = ",".join(str(int(x)) for x in av_pl_levels)

            #Go through all variables
            dv = dataset.variables.keys()
            dv_shape = [dataset.variables[d].shape for d in dv]    #save var shape
            dv_dim = [dataset.variables[d].dimensions for d in dv] #save var dimensions / what it depends on
            varlist = list(zip(dv_shape,dv_dim))
            varframe = pd.DataFrame(varlist, index = dv,columns=["shape", "dim"])

            df.loc[i,"var"] = [varframe.to_dict(orient='index')]
            df.loc[i,"dim"] = [dimframe.to_dict(orient='index')]
            #df.loc[i, "dim"]["pressure"]["value"] = av_pl_levels if df.loc[i,"p_levels"] == True else None

            i+=1

        file_withparam = filter_param( df.copy(), param)
        logging.info("file_param")
        logging.info(file_withparam)

        file_corrtype = filter_type(df.copy(), mbrs,levtype, self.p_level,self.m_level)
        logging.info("file_type")
        logging.info(file_corrtype)

        #file = file_withparam.assign( result=file_withparam['File'].isin(file_corrtype['File']).astype(int))
        file = file_withparam[file_withparam.File.isin(file_corrtype.File)]
        file.reset_index(inplace=True, drop = True)

        file = filter_step(file,self.maxstep)

        #file = file

        logging.info("file")
        #file = filter_any(file)
        logging.info(file)
        if len(file) ==0:#SomeError(ValueError, f'Type not found: choices:{levtype}')
            SomeError( ValueError,  f"File does noes not exist at date: {self.date} for model {self.model} and. Try again (You might want to split up the request). Available files are: \n {df}")
        return file

#check_available(YYYY,MM,DD,HH, "temp", 0)
