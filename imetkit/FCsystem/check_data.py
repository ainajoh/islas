########################################################################
# File name: check_data.py
# This file is part of: FCsystem
#
# LICENSE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.
#
########################################################################
from requests import get
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import logging
from collections import Counter


logging.basicConfig(filename="get_data.log", level = logging.INFO, format = '%(levelname)s : %(message)s')

def SomeError( exception = Exception, message = "Something did not go well" ):
    logging.error(exception(message))
    #source: https://softwareengineering.stackexchange.com/questions/222586/how-should-you-cleanly-restrict-object-property-types-and-values-in-python
    if isinstance( exception.args, tuple ):
        raise exception
    else:
        raise exception(message)

def filter_param(file,param):
    if param:
        for i in range(0, len(file)):
            param_bool = np.array([key in file.loc[i,"var"].keys() for key in param])
            if all(param_bool) == False:
                file.drop([i], inplace=True)
    file.reset_index(inplace=True, drop=True)
    logging.info(file)
    return file

def filter_type(file,mbrs,levtype):
    # secondfilter
    if mbrs != 0 and mbrs != None:
        file = file[file["mbr_bool"] == True]
    if levtype == "ml":
        file = file[file["ml_bool"] == True]
    elif levtype == "pl":
        file = file[file["pl_bool"] == True]

    # third filter, what to choose when we have all options
    #if len(file) > 1:
    #    if mbrs == 0:  # choose the determenistic, smallest one.
    #        file = file[file["mbr_bool"] == False]

    file.reset_index(inplace=True, drop=True)
    return file

def filter_any(file):
    if len(file) > 1:  # want to end up with only one file.
        file = file[0]
        file.reset_index(inplace=True, drop=True)
    return file


class check_data():
    def __init__(self, model, date = None,param=None, mbrs=None,levtype=None, file = None, numbervar = 100, search = None):
        self.date = date
        self.model = model
        self.param = param
        self.mbrs = mbrs
        self.levtype = levtype
        self.file = file
        self.numbervar = numbervar
        self.search = search

        if date != None:
            self.file = self.check_files(date, model, param,  mbrs,levtype)
        if self.param == None and self.date != None:
            self.param = self.check_variable(self.file, self.search)

        if self.date == None and self.param ==None:
            self.param = self.check_variable_all(self.model, self.numbervar, self.search)
            self.date = self.check_available_date(self.model)

    def check_available_date(self, model):
        df = pd.read_csv(f"bin/{model}_filesandvar.csv")
        dfc = df.copy()  # df['base_name'] = [re.sub(r'_[0-9]*T[0-9]*Z.nc','', str(x)) for x in df['File']]
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk"]
        df_base = pd.DataFrame([re.sub(r'_[0-9]*T[0-9]*Z.nc', '', str(x)) for x in df['File']], columns=["base_name"])
        dfc["base_name"] = df_base["base_name"]
        dfc = dfc[~dfc["base_name"].str.contains('|'.join(drop_files))]  # (drop_files)])
        dfc.reset_index(inplace=True, drop=True)
        #df_base = dfc['var'].str.replace(" ", "").str.split(",")  # , expand = True)
        dateti = dfc[["Date","Hour"]].copy()
        dateti.drop_duplicates(keep='first', inplace=True)
        dateti.reset_index(inplace=True, drop=True)

        #print(dfc)
        return dateti

    def check_filecontainingvar(self, model, numbervar, search ):
        #NOT IN USE
        #Nice to have a list of the files containing that var, but that means scraping the web too often.
        #Maybe add on file. scraping only new dates...DID It! Just need to update this function to find file containing: Then another function saying at what date.

        # Todo: update R scripts to only add new lines in filevar info
        df = pd.read_csv(f"bin/{model}_filesandvar.csv")
        dfc = df.copy()
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk"]
        df_base = pd.DataFrame([re.sub(r'_[0-9]*T[0-9]*Z.nc', '', str(x)) for x in df['File']], columns=["base_name"])
        dfc["base_name"] = df_base["base_name"]
        dfc = dfc[~dfc["base_name"].str.contains('|'.join(drop_files))]  # (drop_files)])

        df_base = dfc['var'].str.replace(" ", "").str.split(",")  # , expand = True)
        search = "wind"
        #param = df_base[df_base.apply(lambda x: x.str.contains(search)).any(axis=1)]
        test = ["heipadeg", "du", "hei du"]
        print("hei" in test)
        print( [ df_base["pressure" in s] for s in df_base] ) #s.isin(['a'])
        flattened = [val for sublist in df_base[:] for val in sublist]

        print(param)


    def check_variable_all(self, model, numbervar, search ):
        df = pd.read_csv(f"bin/{model}_filesandvar.csv")
        dfc = df.copy()  # df['base_name'] = [re.sub(r'_[0-9]*T[0-9]*Z.nc','', str(x)) for x in df['File']]
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk"]
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
        #Find what files exist at that date

        page = get(base_url + YYYY+"/"+MM+"/"+DD+ "/catalog.html")
        soup = BeautifulSoup(page.text, 'html.parser')
        rawfiles= soup.table.find_all("a")
        ff =[i.text for i in rawfiles]
        pattern=re.compile(f'.*{YYYY}{MM}{DD}T{HH}Z.nc')
        ff= pd.DataFrame( data = list(filter(pattern.match,ff)), columns=["File"])
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk"]
        df = ff.copy()[~ff["File"].str.contains('|'.join(drop_files))] #(drop_files)])

        df.reset_index(inplace=True, drop = True)
        df["var"] = None
        df["dim"] = None
        df["mbr_bool"] = False
        df["ml_bool"] = False
        df["pl_bool"] = False
        i=0
        while i<len(df):
            file=df["File"][i]
            dataset = Dataset(base_urlfile + YYYY+"/"+MM+"/"+DD+ "/"+ file)
            dn = dataset.dimensions.keys()
            ds = [dataset.dimensions[d].size for d in dn  ]
            dimdic = dict(zip(dn,ds))

            df.loc[i,"mbr_bool"] = ("ensemble_member" in dimdic) and (dimdic["ensemble_member"]>=10)
            df.loc[i,"ml_bool"] = ("hybrid" in dimdic) and (dimdic["hybrid"]>=65)
            df.loc[i,"pl_bool"] = ("pressure" in dimdic) and (dimdic["pressure"]>=10)

            dv = dataset.variables.keys()
            dvs = [dataset.variables[d].shape for d in dv  ]
            vardic = dict(zip(dv, dvs))
            #print(vardic.keys())
            df.loc[i,"var"] = [vardic]
            df.loc[i,"dim"] = [dimdic]
            i+=1



        file_withparam = filter_param( df.copy(), param)
        logging.info("file_param")
        logging.info(file_withparam)

        file_corrtype = filter_type(df.copy(), mbrs,levtype)
        logging.info("file_type")
        logging.info(file_corrtype)

        #file = file_withparam.assign( result=file_withparam['File'].isin(file_corrtype['File']).astype(int))
        file = file_withparam[file_withparam.File.isin(file_corrtype.File)]
        file.reset_index(inplace=True, drop = True)

        logging.info("file")
        #file = filter_any(file)
        logging.info(file)
        if len(file) ==0:#SomeError(ValueError, f'Type not found: choices:{levtype}')
            SomeError( ValueError,  f"File does noes not exist at date: {self.date} for model {self.model} and. Try again (You might want to split up the request). Available files are: \n {df}")
        return file

#check_available(YYYY,MM,DD,HH, "temp", 0)
