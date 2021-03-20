from weathervis.config import *
from weathervis.domain import *
from weathervis.utils import *
from weathervis.check_data import *
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
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import sys
import matplotlib.patheffects as pe
#from cartopy.io import shapereader  # For reading shapefiles containg high-resolution coastline.
from copy import deepcopy
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl

import itertools

# if len(param_we_want_that_areNOT_available) != 0:
#        print(f"The requested parameters are not all available. Missing: {param_we_want_that_areNOT_available}")
#            raise ValueError
#            break
def domain_input_handler(dt, model, domain_name, domain_lonlat, file, point_name):
  #print(point_name)
  #print(domain_name)
  #print(domain_lonlat)
  if domain_name or domain_lonlat:
    if domain_lonlat:
      print(f"\n####### Setting up domain for coordinates: {domain_lonlat} ##########")
      data_domain = domain(dt, model, file=file, lonlat=domain_lonlat)
    else:
      data_domain = domain(dt, model, file=file)

    if domain_name != None and domain_name in dir(data_domain):
      print(f"\n####### Setting up domain: {domain_name} ##########")
      domain_name = domain_name.strip()
      #data_domain = domain(dt, model, file=file, domain_name=domain_name)
      if re.search("\(\)$", domain_name):
        func = f"data_domain.{domain_name}"
      else:
        func = f"data_domain.{domain_name}()"
      print(func)
      print(domain_name)
      eval(func)
    else:
      print(f"No domain found with that name; {domain_name}")
  else:
    data_domain=None
  if (point_name !=None and domain_name == None and domain_lonlat == None):
     print("GGGGGOOOO")
     data_domain = domain(dt, model, file=file, point_name=point_name)
     print("DOM DONE")
  print(data_domain)
  return data_domain


def find_best_combinationoffiles(all_param,fileobj,m_level=None,p_level=None):    #how many ways ca we split up all the possible balls between these kids?
    m_level = 60 if m_level is None else max(m_level)
    filenames = []#all_balls
    tot_param_we_want_that_are_available = []
    tot_param_we_want_that_areNOT_available = []
    for i in range(0, len(fileobj)):
        param_available= [*fileobj.loc[i].loc["var"].keys()]
        param_we_want_that_are_available = [x for x in param_available if x in all_param]
        param_we_want_that_areNOT_available = [x for x in all_param if x not in param_we_want_that_are_available]
        tot_param_we_want_that_areNOT_available += [param_we_want_that_areNOT_available]
        tot_param_we_want_that_are_available+= [param_we_want_that_are_available]
        filenames += [fileobj.loc[i].loc["File"]]

    #getting the unique flattened version of the total parameter that was available and that was not.
    tot_unique_avalable = []
    for sublist in tot_param_we_want_that_are_available:
        for item in sublist:
            if item not in tot_unique_avalable:
                tot_unique_avalable.append(item)
    tot_NOTunique_avalable = []
    for sublist in tot_param_we_want_that_areNOT_available:
        for item in sublist:
            if item not in tot_NOTunique_avalable:
                tot_NOTunique_avalable.append(item)
    #Contains the parameters not found in any file.
    bad_param = [x for x in tot_NOTunique_avalable if x not in tot_unique_avalable]
    #if bad_param:

    #UNCOMMENT IF YOU WANT IT TO STOP WHEN PARAM YOU WANT IS NOT FOUND AT ALL
    #if len(not_available_at_all) != 0:
    #    print(f"The requested parameters are not all available. Missing: {not_available_at_all}")
    #    raise ValueError #what if we set these variables to None such that no error eill occur with plotting, only blanks
    config_overrides_r = dict(zip(filenames, tot_param_we_want_that_are_available))

    def filer_param_by_modellevels(config_overrides_r,tot_param_we_want_that_are_available):
        print(len(fileobj))
        for i in range(0,len(fileobj)):
            thisfileobj = fileobj.loc[i]
            varname = tot_param_we_want_that_are_available[i]
            if len(varname) == 0: #jump over file if no parameter needed in it
                continue
            var = pd.DataFrame.from_dict(thisfileobj.loc["var"], orient="index")
            varname = [varname] if type(varname) is not list else varname
            pandas_df = var.loc[varname]
            f = pandas_df[pandas_df.dim.astype(str).str.contains("hybrid")] #keep only ml variables.

            if len(f) != 0: #if var depends on hybrid.
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
    possible_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)][:10]
    ppd = pd.DataFrame([], columns=["file", "keys", "len", "combo"])
    ppd.sort_values(by='len', inplace=True)

    iii = 0
    #This for loop takes time if it is many combinations, therefore reduces combinations.
    # bUT IT CAN BLE glitchy because I CAN NOT sort it based on how many files, but seems python might do this automatically..
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
    print(ppd)
    return ppd,bad_param

def retrievenow(our_choice,model,step, date,fileobj,m_level, domain_name=None, domain_lonlat=None,bad_param=[],bad_param_sfx=[],point_name=None):
    print("HEEEE")
    fixed_var = ["ap", "b", "ap2", "b2", "pressure", "hybrid", "hybrid2","hybrid0"]  # this should be gotten from get_data
    #indexidct = {"time": step, "y": y, "x": x, "ensemble_member": mbrs,
    #             "pressure": pl_idx, "hybrid": m_level, "hybrid2": m_level, "hybrid0": non,
    #             "height0": non, "height1": non, "height2": non,
    #             "height3": non, "height7": non, "height6": non, 'height_above_msl': non, "mean_sea_level": non,
    #             "atmosphere_as_single_layer": non}
    ap_prev = 0
    b_prev = 0
    ap2_prev = 0
    b2_prev = 0

    for i in range(0,len(our_choice.file)):
        ourfilename = our_choice.file[i]
        combo = our_choice.combo
        ourparam = [k for k, v in combo.items() if v == ourfilename]
        ourfileobj = fileobj[fileobj["File"].isin([ourfilename])]
        ourfileobj.reset_index(inplace=True, drop=True)
        print("data_domain")
        print(ourfileobj)
        data_domain = domain_input_handler(dt=date, model=model, domain_name=domain_name, domain_lonlat=domain_lonlat, file =ourfileobj, point_name=point_name)
        print("retrieve strt")
        print(data_domain)
        dmet = get_data(model=model, param=ourparam, file=ourfileobj, step=step, date=date,m_level=m_level,data_domain=data_domain)
        print("real retriete")
        print(dmet.url)
        dmet.retrieve()
        print("retriete done ")
        #for pm in dmet_new:
        if i >= 1: #sec run
            print("stat merging objects")
            for pm in dmet_old.param:
                #The if statements should be done more auto, maybe a dictionary.
                if pm in fixed_var:
                    ap_prev = len(getattr(dmet_old, pm)) if pm in dmet_old.param else 0
                    ap_next = len(getattr(dmet, pm)) if pm in dmet.param else 0
                    if ap_next > ap_prev:  # if next is bigger dont overwrite with old one
                        continue
                setattr(dmet, pm, getattr(dmet_old, pm))
            print("done objects")
        #add unit later
        dmet_old = dmet
    for bparam in bad_param:
        setattr(dmet, bparam, None)

    good_sfx = np.setdiff1d(["SFX_"+b for b in bad_param],bad_param_sfx)
    print(good_sfx)
    for gprmsfx in good_sfx:
        gprm = gprmsfx.replace("SFX_","")
        setattr(dmet, gprm, getattr(dmet,gprmsfx))





    return dmet, data_domain,bad_param

def checkget_data_handler(all_param,date,  model, step, p_level= None, m_level=None, mbrs=None, domain_name=None, domain_lonlat=None, point_name=None):
    fileobj = check_data(model, date=date, step=step).file
    print(fileobj)
    print(all_param)
    print("start finding choices")
    all_choices, bad_param  = find_best_combinationoffiles(all_param=all_param, fileobj=fileobj,m_level=m_level,p_level=p_level)
    bad_param_sfx=[]
    if bad_param:
        new_bad = ["SFX_"+x for x in bad_param]
        all_param = all_param + new_bad
        all_choices, bad_param_sfx = find_best_combinationoffiles(all_param=all_param, fileobj=fileobj, m_level=m_level,
                                                              p_level=p_level)
        print("bad_param")
        print(bad_param)
        print("bad_param_sfx")
        print(bad_param_sfx)
        #Ass SFX_param to it and try again.

    print(all_choices)
    print("stopped finding choices")

    # RETRIEVE FROM THE BEST COMBINATIONS AND TOWARDS WORSE COMBINATION IF ANY ERROR
    for i in range(0, len(all_choices)):

        try:
            print("getting data")#our_choice,model,step, date,fileobj,m_level, domain_name=None, domain_lonlat=None
            dmet,data_domain,bad_param = retrievenow(our_choice = all_choices.loc[i],model=model,step=step, date=date,fileobj=fileobj,
                               m_level=m_level,domain_name=domain_name, domain_lonlat=domain_lonlat, bad_param = bad_param,bad_param_sfx = bad_param_sfx,point_name=point_name)
            break
        except:
            #del (dmet)
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Next entry.")
            print(" ")
    #for i in range(0,bad_param):

    return dmet,data_domain,bad_param

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", default=None, nargs="+")
    parser.add_argument("--steps", default=[0, 10], nargs="+", type=int,
                        help="forecast times example --steps 0 3 gives time 0 to 3")
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument("--domain_name", default=None, help="see domain.py")
    parser.add_argument("--domain_lonlat", default=None, nargs="+", type=float, help="lonmin lonmax latmin latmax")
    parser.add_argument("--param_all", default=None, nargs="+", type=str)
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

    fileobj = check_data(args.model, date=str(args.datetime[0]), step=args.steps).file
    all_choices, bad_param = find_best_combinationoffiles(all_param, fileobj)

    #RETRIEVE FROM THE BEST COMBINATIONS AND TOWARDS WORSE COMBINATION IF ANY ERROR
    for i in range(0, len(all_choices)):
        try:
            dmet = retrievenow(all_choices.loc[i],args.model,args.steps, str(args.datetime[0]))
            break
        except:
            #del(dmet)
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Next entry.")
            print(" ")
