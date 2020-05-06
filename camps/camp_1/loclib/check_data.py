from urllib.request import urlopen
from requests import get
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import logging

def filter_param(file,param):
    for i in range(0, len(file)):
        param_bool = np.array([key in file.loc[i].at["var"].keys() for key in param])
        if all(param_bool) == False:
            file.drop([i], inplace=True)
            # print(self.param[ ~param_bool ] )
    file.reset_index(inplace=True, drop=True)
    logging.info(file)
    return file

def filter_type(file,mbrs,levtype):
    # secondfilter
    if mbrs != 0:
        file = file[file["mbr_bool"] == True]
    if levtype == "ml":
        file = file[file["ml_bool"] == True]
    elif levtype == "pl":
        file = file[file["pl_bool"] == True]
    # third filter, what to choose when we have all options
    if len(file) > 1:
        if mbrs == 0:  # choose the determenistic, smallest one.
            file = file[file["mbr_bool"] == False]

    file.reset_index(inplace=True, drop=True)
    logging.info(file)
    return file

def filter_any(file):
    if len(file) > 1:  # want to end up with only one file.
        file = file[0]
        file.reset_index(inplace=True, drop=True)
    return file

def check_available(date, mbrs,levtype, param, model):
    YYYY = date[0:4]
    MM = date[4:6]
    DD = date[6:8]
    HH = date[8:10]

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
    df["var"] =None
    df["dim"] = None
    df["mbr_bool"] = None
    df["ml_bool"] = None
    df["pl_bool"] = None
    i=0
    while i<len(df):
        file=df["File"][i]
        dataset = Dataset(base_urlfile + YYYY+"/"+MM+"/"+DD+ "/"+ file)
        dn = dataset.dimensions.keys()
        ds = [dataset.dimensions[d].size for d in dn  ]
        dimdic = dict(zip(dn,ds))
        df.loc[i].at["mbr_bool"] = ("ensemble_member" in dimdic) and (dimdic["ensemble_member"]>=10)
        df.loc[i].at["ml_bool"] = ("hybrid" in dimdic) and (dimdic["hybrid"]>=65)
        df.loc[i].at["pl_bool"] = ("pressure" in dimdic) and (dimdic["pressure"]>=10)

        dv = dataset.variables.keys()
        dvs = [dataset.variables[d].shape for d in dv  ]
        vardic = dict(zip(dv, dvs))
        df.loc[i].at["var"] = vardic
        df.loc[i].at["dim"] = dimdic
        i+=1

    file_withparam = filter_param(df.copy(),param)
    file_corrtype = filter_type(df.copy(), mbrs,levtype)
    file = file_withparam.assign(result=file_withparam['File'].isin(file_withparam['File']).astype(int))

    #file = filter_any(file)
    logging.info("file")

    return df, file

#check_available(YYYY,MM,DD,HH, "temp", 0)
