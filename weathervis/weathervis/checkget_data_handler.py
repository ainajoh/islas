from weathervis.config import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.calculation import *
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import matplotlib.patheffects as pe
from cartopy.io import shapereader  # For reading shapefiles containg high-resolution coastline.
from copy import deepcopy
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl

import itertools

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def find_best_combinationoffiles(all_param,fileobj,m_level=None,p_level=None):    #how many ways ca we split up all the possible balls between these kids?
    m_level = 60 if m_level is None else max(m_level)
    filenames = []#all_balls
    tot_param_we_want_that_are_available = []
    for i in range(0, len(fileobj)):
        param_available= [*fileobj.loc[i].loc["var"].keys()]
        param_we_want_that_are_available = [x for x in param_available if x in all_param]
        tot_param_we_want_that_are_available+= [param_we_want_that_are_available]
        filenames += [fileobj.loc[i].loc["File"]]
    config_overrides_r = dict(zip(filenames, tot_param_we_want_that_are_available))
    print("eTOOOTeee")
    print(tot_param_we_want_that_are_available)
    def filer_param_by_modellevels(config_overrides_r,tot_param_we_want_that_are_available):
        print(len(fileobj))
        for i in range(0,len(fileobj)):
            thisfileobj = fileobj.loc[i]
            varname = tot_param_we_want_that_are_available[i]
            if len(varname) == 0:
                continue
            var = pd.DataFrame.from_dict(thisfileobj.loc["var"], orient="index")
            pandas_df = var.loc[varname]
            f = pandas_df[pandas_df.dim.astype(str).str.contains("hybrid")] #keep only ml variables.
            if len(f) != 0:
                dimen = [f.apply(lambda row: dict(zip(row['dim'],row['shape'])), axis=1)][0]#.loc["dim"]
                dimofmodellevel = [dimen.apply(lambda row: [value for key, value in row.items() if 'hybrid' in key.lower()])][0]#.loc["dim
                removethese = dimofmodellevel[dimofmodellevel.apply(lambda row: row[0]<m_level)]#.loc["dim"]
                val = [*removethese.index]
                key = thisfileobj["File"]
                config_overrides_r[key] = [x for x in config_overrides_r[key] if x not in val]
            return config_overrides_r

    config_overrides_r = filer_param_by_modellevels(config_overrides_r,tot_param_we_want_that_are_available)
    print(config_overrides_r)
    #flip it
    config_overrides = {}
    for key, value in config_overrides_r.items():
         for prm in value:
             config_overrides.setdefault(prm, []).append(key)

    keys, values = zip(*config_overrides.items())
    possible_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    ppd = pd.DataFrame([], columns=["file", "keys", "len", "combo"])
    iii = 0
    for combination in possible_combinations:
        filesincombination = [*combination.values()]
        uniquelistoffiles = list(set(filesincombination))
        ppd.at[iii, 'file'] = uniquelistoffiles
        ppd.at[iii, 'keys'] = iii
        ppd.at[iii, 'len'] = len(uniquelistoffiles)
        ppd.at[iii, 'combo'] = combination
        iii += 1
    ppd.sort_values(by='len', inplace=True)
    ppd.reset_index(inplace=True)
    our_choice = ppd.loc[0] #The best combination retrieving from least amount of files.
    return ppd

def retrievenow(our_choice):
    for i in range(0,len(our_choice.file)):
        ourfilename = our_choice.file[i]
        combo = our_choice.combo
        ourparam = [k for k, v in combo.items() if v == ourfilename]
        ourfileobj = fileobj[fileobj["File"].isin([ourfilename])]
        ourfileobj.reset_index(inplace=True, drop=True)
        dmet = get_data(model=model, param=ourparam, file=ourfileobj, step=step, date=date)
        dmet.retrieve()
        if i >= 1: #sec run
            for pm in dmet_old.param:
                setattr(dmet, pm, getattr(dmet_old, pm))
                #add unit later
        dmet_old = dmet
    return dmet

def checkget_data_handler(all_param, model, date, step, mbrs=None,levtype=None, p_level= None, m_level=None, file = None):

    fileobj = check_data(model, date=date, step=step).file
    print(fileobj)
    print(all_param)
    all_choices = find_best_combinationoffiles(all_param=all_param, fileobj=fileobj,m_level=m_level,p_level=p_level)
    print(all_choices)
    # RETRIEVE FROM THE BEST COMBINATIONS AND TOWARDS WORSE COMBINATION IF ANY ERROR
    for i in range(0, len(all_choices)):
        try:
            dmet = retrievenow(all_choices.loc[i])
            break
        except:
            del (dmet)
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Next entry.")
            print(" ")
    return dmet

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", default=None, nargs="+")
    parser.add_argument("--steps", default=[0, 10], nargs="+", type=int,
                        help="forecast times example --steps 0 3 gives time 0 to 3")
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument("--domain_name", default=None, help="see domain.py")
    parser.add_argument("--domain_lonlat", default=None, nargs="+", type=float, help="lonmin lonmax latmin latmax")
    parser.add_argument("--param_all", default=None, nargs="+", type=string)
    parser.add_argument("--point_name", default=None, help="see sites.csv")
    parser.add_argument("--point_lonlat", default=None, nargs="+", type=float, help="lon lat")
    parser.add_argument("--point_num", default=1, type=int)
    parser.add_argument("--plot", default="all", help="Display legend")
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()

    param_pl = []
    param_ml = ["air_temperature_ml", "specific_humidity_ml"]
    param_sfc = ["surface_air_pressure", "air_pressure_at_sea_level", "air_temperature_0m", "air_temperature_2m",
                 "relative_humidity_2m", "x_wind_gust_10m", "y_wind_gust_10m", "x_wind_10m", "y_wind_10m",
                 "specific_humidity_2m", "precipitation_amount_acc", "convective_cloud_area_fraction",
                 "cloud_area_fraction", "high_type_cloud_area_fraction", "medium_type_cloud_area_fraction",
                 "low_type_cloud_area_fraction", "rainfall_amount", "snowfall_amount", "graupelfall_amount",
                 "land_area_fraction"]
    param_sfc = ["specific_humidity_2m"]
    all_param = param_sfc + param_ml + param_pl

    fileobj = check_data(model, date=date, step=step).file
    all_choices = find_best_combinationoffiles(all_param, fileobj)

    #RETRIEVE FROM THE BEST COMBINATIONS AND TOWARDS WORSE COMBINATION IF ANY ERROR
    for i in range(0, len(all_choices)):
        try:
            dmet = retrievenow(all_choices.loc[i])
            break
        except:
            del(dmet)
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Next entry.")
            print(" ")
