from weathervis.config import *
from weathervis.utils import *
from weathervis.check_data import *
from weathervis.domain import *
from weathervis.calculation import *
from weathervis.get_data import *
from weathervis.checkget_data_handler import *

import platform
import os

def domain_input_handler(dt, model, domain_name, domain_lonlat, file):
  if domain_name or domain_lonlat:
    if domain_lonlat:
      print(f"\n####### Setting up domain for coordinates: {domain_lonlat} ##########")
      data_domain = domain(dt, model, file=file, lonlat=domain_lonlat)
    else:
      data_domain = domain(dt, model, file=file)

    if domain_name != None and domain_name in dir(data_domain):
      print(f"\n####### Setting up domain: {domain_name} ##########")
      domain_name = domain_name.strip()
      if re.search("\(\)$", domain_name):
        func = f"data_domain.{domain_name}"
      else:
        func = f"data_domain.{domain_name}()"
      eval(func)
    else:
      print(f"No domain found with that name; {domain_name}")
  else:
    data_domain=None
  return data_domain

def point(datetime, point_name, point_lonlat,number_grid=1):
    if point_name:
        if "cyclone.hpc.uib.no" in platform.node():
            print("detected cyclone")
            sites = pd.read_csv("../githubclones/islas/weathervis/weathervis/data/sites.csv", sep=";", header=0, index_col=0)
        else:
            sites = pd.read_csv("../../weathervis/weathervis/data/sites.csv", sep=";", header=0, index_col=0)
        point_lonlat = [sites.loc[point_name].lon, sites.loc[point_name].lat]
        print(point_lonlat)

    #test nearest lonlat
    param_sfc = ["air_pressure_at_sea_level"]
    dmap_meps, dom_name, bad_param = checkget_data_handler(param_sfc, date=datetime, model="AromeArctic", step=[0,1])
    
    #check_all = check_data(date=datetime, model="AromeArctic", param=["longitude"], step=1)
    #file_all = check_all.file
    #print()
    #data_domain = domain_input_handler(dt=datetime, model="AromeArctic", domain_name="AromeArctic", file=file_all,domain_lonlat=None)
    #dmap_meps = get_data(model="AromeArctic", data_domain=data_domain, param=["longitude"], file=file_all, step=1,
    #                     date=datetime)
    #dmap_meps.retrieve()
    print("NUMBER GRID")
    print(number_grid)
    closest_idx= nearest_neighbour_idx(point_lonlat[0], point_lonlat[1], dmap_meps.longitude, dmap_meps.latitude, nmin=number_grid)

    x_llc=np.min(closest_idx[0])
    y_llc = np.min(closest_idx[1])
    x_urc = np.max(closest_idx[0])
    y_urc = np.max(closest_idx[1])
    #print(x_llc*2500)
    #print(y_llc * 2500)
    #print(x_urc * 2500)
    #print(y_urc * 2500)
    #return [x_llc*2500,y_llc * 2500, x_urc * 2500, y_urc * 2500]
    return [y_llc*2500,x_llc * 2500, y_urc * 2500, x_urc * 2500]

def area(datetime, domain_name, domain_lonlat):
    check_all = check_data(date=datetime, model="AromeArctic", step=0)
    file_all = check_all.file
    data_domain = domain_input_handler(dt=datetime, model="AromeArctic", domain_name=domain_name, domain_lonlat=domain_lonlat,
                                       file=file_all)
    iii = data_domain.idx
    xllc = min(iii[0])
    xurc = np.max(iii[0])
    yllc = min(*iii[1])
    yurc = max(iii[1])

    return [yllc * 2500, xllc * 2500, yurc * 2500, xurc * 2500]
def make_inputfile(file, datetime, steps, domain_name, domain_lonlat, point_name, point_lonlat, number_grid,
     begin_YYYYMMDD, end_YYYYMMDD,
     begin_HHMMSS, end_HHMMSS,
     begin_rel_YYYYMMDD="", 
     begin_rel_HHMMSS="",
     sim_direction=1, ZPOINT_1=950,ZPOINT_2=800,rel_numb=1,end_rel_YYYYMMDD="",end_rel_HHMMSS="",
     rel1_ZTYPE="3",rel1_NUMB_PART="50000",rel1_XMASS="1",
     begin_rel2_YYYYMMDD="", end_rel2_YYYYMMDD="",
     begin_rel2_HHMMSS="", end_rel2_HHMMSS="",
     rel2_sim_direction="", rel2_ZPOINT_1="",rel2_ZPOINT_2="", rel2_dom=["","","",""],rel2_dom_name="",
     rel2_ZTYPE="", rel2_NUMB_PART="",rel2_XMASS="",
     begin_rel3_YYYYMMDD="", end_rel3_YYYYMMDD="",
     begin_rel3_HHMMSS="", end_rel3_HHMMSS="",
     rel3_sim_direction="", rel3_ZPOINT_1="",rel3_ZPOINT_2="", rel3_dom=["","","",""], rel3_dom_name="",
     rel3_ZTYPE="", rel3_NUMB_PART="",rel3_XMASS="",
     begin_rel4_YYYYMMDD="", end_rel4_YYYYMMDD="",
     begin_rel4_HHMMSS="", end_rel4_HHMMSS="",
     rel4_sim_direction="", rel4_ZPOINT_1="",rel4_ZPOINT_2="", rel4_dom=["","","",""],rel4_dom_name="",
     rel4_ZTYPE="", rel4_NUMB_PART="", rel4_XMASS="",
     begin_rel5_YYYYMMDD="", end_rel5_YYYYMMDD="",
     begin_rel5_HHMMSS="", end_rel5_HHMMSS="",
     rel5_sim_direction="", rel5_ZPOINT_1="", rel5_ZPOINT_2="", rel5_dom=["", "", "", ""],
     rel5_dom_name="",
     rel5_ZTYPE="", rel5_NUMB_PART="", rel5_XMASS="",
     begin_rel6_YYYYMMDD="", end_rel6_YYYYMMDD="",
     begin_rel6_HHMMSS="", end_rel6_HHMMSS="",
     rel6_sim_direction="", rel6_ZPOINT_1="", rel6_ZPOINT_2="", rel6_dom=["", "", "", ""],
     rel6_dom_name="",
     rel6_ZTYPE="", rel6_NUMB_PART="", rel6_XMASS=""):

    #rel1_ZTYPE; 1 for m above ground, 2 for m above sea level, 3 pressure
    dom_name="lonlat"
    if point_name or point_lonlat:
        dom = point(datetime, point_name, point_lonlat, number_grid)
        if point_name:
            dom_name= point_name
    elif domain_name or domain_lonlat:
        dom = area(datetime, domain_name, domain_lonlat)
        if domain_name:
            dom_name= domain_name
    #file="/Users/ainajoh/flexarome.input-ISLAS-lowres__INPUT"

    if int(sim_direction)==-1:
        #rel 1
        new_beg=end_YYYYMMDD
        new_end=begin_YYYYMMDD
        begin_YYYYMMDD=new_beg
        end_YYYYMMDD=new_end
        #rel2 todo: add more


    if begin_rel_YYYYMMDD=="":
        begin_rel_YYYYMMDD = begin_YYYYMMDD
    if begin_rel_HHMMSS=="":
        begin_rel_HHMMSS = begin_HHMMSS
    
    if end_rel_YYYYMMDD =="":
        end_rel_YYYYMMDD = begin_rel_YYYYMMDD
        print(end_rel_YYYYMMDD)
    if end_rel_HHMMSS =="":
        end_rel_HHMMSS = begin_rel_HHMMSS
        
    
    if begin_rel2_YYYYMMDD != "":
        rel_numb += 1
        rel2_sim_direction=sim_direction
        rel2_dom=dom
        rel2_XMASS=rel1_XMASS
        rel2_NUMB_PART = rel1_NUMB_PART
        rel2_ZTYPE = rel1_ZTYPE
        rel2_dom_name = dom_name+"rel2"
        if rel2_ZPOINT_1 =="":
            rel2_ZPOINT_1 = ZPOINT_1
            rel2_ZPOINT_2 = ZPOINT_2
        if end_rel2_YYYYMMDD =="":
            end_rel2_YYYYMMDD = begin_rel2_YYYYMMDD
        if end_rel2_HHMMSS =="":
            end_rel2_HHMMSS = begin_rel2_HHMMSS

    if begin_rel3_YYYYMMDD != "":
        rel_numb += 1
        rel3_sim_direction=sim_direction
        rel3_dom=dom
        rel3_XMASS=rel1_XMASS
        rel3_NUMB_PART=rel1_NUMB_PART
        rel3_dom_name = dom_name + "rel3"
        rel3_ZTYPE = rel1_ZTYPE

        if rel3_ZPOINT_1 == "":
            rel3_ZPOINT_1 = ZPOINT_1
            rel3_ZPOINT_2 = ZPOINT_2
        if end_rel3_YYYYMMDD =="":
            end_rel3_YYYYMMDD = begin_rel3_YYYYMMDD
        if end_rel3_HHMMSS =="":
            end_rel3_HHMMSS = begin_rel3_HHMMSS
    if begin_rel4_YYYYMMDD != "":
        rel_numb += 1
        rel4_sim_direction=sim_direction
        rel4_dom=dom
        rel4_XMASS = rel1_XMASS
        rel4_NUMB_PART=rel1_NUMB_PART
        rel4_dom_name = dom_name + "rel4"
        rel4_ZTYPE = rel1_ZTYPE

        if rel4_ZPOINT_1 == "":
            rel4_ZPOINT_1 = ZPOINT_1
            rel4_ZPOINT_2 = ZPOINT_2
        if end_rel4_YYYYMMDD =="":
            end_rel4_YYYYMMDD = begin_rel4_YYYYMMDD
        if end_rel4_HHMMSS =="":
            end_rel4_HHMMSS = begin_rel4_HHMMSS
    if begin_rel5_YYYYMMDD != "":
        rel_numb += 1
        rel5_sim_direction=sim_direction
        rel5_dom=dom
        rel5_XMASS = rel1_XMASS
        rel5_NUMB_PART=rel1_NUMB_PART
        rel5_dom_name = dom_name + "rel4"
        rel5_ZTYPE = rel1_ZTYPE

        if rel5_ZPOINT_1 == "":
            rel5_ZPOINT_1 = ZPOINT_1
            rel5_ZPOINT_2 = ZPOINT_2
        if end_rel5_YYYYMMDD =="":
            end_rel5_YYYYMMDD = begin_rel5_YYYYMMDD
        if end_rel5_HHMMSS =="":
            end_rel5_HHMMSS = begin_rel5_HHMMSS
    if begin_rel6_YYYYMMDD != "":
        rel_numb += 1
        rel6_sim_direction=sim_direction
        rel6_dom=dom
        rel6_XMASS = rel1_XMASS
        rel6_NUMB_PART=rel1_NUMB_PART
        rel6_dom_name = dom_name + "rel4"
        rel6_ZTYPE = rel1_ZTYPE

        if rel6_ZPOINT_1 == "":
            rel6_ZPOINT_1 = ZPOINT_1
            rel6_ZPOINT_2 = ZPOINT_2
        if end_rel6_YYYYMMDD =="":
            end_rel6_YYYYMMDD = begin_rel6_YYYYMMDD
        if end_rel6_HHMMSS =="":
            end_rel6_HHMMSS = begin_rel6_HHMMSS

    import re
    subdict={"{date_input_for_flexpart}":datetime,
             "{begin_YYYYMMDD}":begin_YYYYMMDD,
             "{end_YYYYMMDD}":end_YYYYMMDD,
             "{begin_HHMMSS}":begin_HHMMSS,
             "{end_HHMMSS}":end_HHMMSS,
             "{rel1_begin_YYYYMMDD}":begin_rel_YYYYMMDD,
             "{rel1_end_YYYYMMDD}":end_rel_YYYYMMDD,
             "{rel1_begin_HHMMSS}":begin_rel_HHMMSS,
             "{rel1_end_HHMMSS}":end_rel_HHMMSS,
             "{sim_direction}":sim_direction,#sim_direction
             "{rel1_min_1}":str(dom[0]),
             "{rel1_min_2}":str(dom[1]),
             "{rel1_max_1}":str(dom[2]),
             "{rel1_max_2}":str(dom[3]),
             "{rel1_ZPOINT_1}": ZPOINT_1,
             "{rel1_ZPOINT_2}": ZPOINT_2,
             "{rel1_domain_name}":dom_name,
             "{rel1_ZTYPE}":rel1_ZTYPE,
             "{rel1_NUMB_PART}":rel1_NUMB_PART,
             "{rel1_XMASS}":rel1_XMASS}

    #if begin_rel2_YYYYMMDD != None:
    subdict_rel2 = {"{rel2_begin_YYYYMMDD}": begin_rel2_YYYYMMDD,
                    "{rel2_end_YYYYMMDD}": end_rel2_YYYYMMDD,
                    "{rel2_begin_HHMMSS}": begin_rel2_HHMMSS,
                    "{rel2_end_HHMMSS}": end_rel2_HHMMSS,
                    "{sim_direction}": rel2_sim_direction,  # sim_direction
                    "{rel2_min_1}": str(rel2_dom[0]),
                    "{rel2_min_2}": str(rel2_dom[1]),
                    "{rel2_max_1}": str(rel2_dom[2]),
                    "{rel2_max_2}": str(rel2_dom[3]),
                    "{rel2_ZPOINT_1}": rel2_ZPOINT_1,
                    "{rel2_ZPOINT_2}": rel2_ZPOINT_2,
                    "{rel2_domain_name}": rel2_dom_name,
                    "{rel2_ZTYPE}":rel2_ZTYPE,
                    "{rel2_NUMB_PART}":rel2_NUMB_PART,
                    "{rel2_XMASS}":rel2_XMASS}
    subdict_rel3 = {"{rel3_begin_YYYYMMDD}": begin_rel3_YYYYMMDD,
                    "{rel3_end_YYYYMMDD}": end_rel3_YYYYMMDD,
                    "{rel3_begin_HHMMSS}": begin_rel3_HHMMSS,
                    "{rel3_end_HHMMSS}": end_rel3_HHMMSS,
                    "{sim_direction}": rel3_sim_direction,  # sim_direction
                    "{rel3_min_1}": str(rel3_dom[0]),
                    "{rel3_min_2}": str(rel3_dom[1]),
                    "{rel3_max_1}": str(rel3_dom[2]),
                    "{rel3_max_2}": str(rel3_dom[3]),
                    "{rel3_ZPOINT_1}": rel3_ZPOINT_1,
                    "{rel3_ZPOINT_2}": rel3_ZPOINT_2,
                    "{rel3_domain_name}": rel3_dom_name,
                    "{rel3_ZTYPE}":rel3_ZTYPE,
                    "{rel3_NUMB_PART}":rel3_NUMB_PART,
                    "{rel3_XMASS}":rel3_XMASS}
    subdict_rel4 = {"{rel4_begin_YYYYMMDD}": begin_rel4_YYYYMMDD,
                    "{rel4_end_YYYYMMDD}": end_rel4_YYYYMMDD,
                    "{rel4_begin_HHMMSS}": begin_rel4_HHMMSS,
                    "{rel4_end_HHMMSS}": end_rel4_HHMMSS,
                    "{sim_direction}": rel4_sim_direction,  # sim_direction
                    "{rel4_min_1}": str(rel4_dom[0]),
                    "{rel4_min_2}": str(rel4_dom[1]),
                    "{rel4_max_1}": str(rel4_dom[2]),
                    "{rel4_max_2}": str(rel4_dom[3]),
                    "{rel4_ZPOINT_1}": rel4_ZPOINT_1,
                    "{rel4_ZPOINT_2}": rel4_ZPOINT_2,
                    "{rel4_domain_name}": rel4_dom_name,
                    "{rel4_ZTYPE}":rel4_ZTYPE,
                    "{rel4_NUMB_PART}":rel4_NUMB_PART,
                    "{rel4_XMASS}":rel4_XMASS}
    subdict_rel5 = {"{rel5_begin_YYYYMMDD}": begin_rel5_YYYYMMDD,
                    "{rel5_end_YYYYMMDD}": end_rel5_YYYYMMDD,
                    "{rel5_begin_HHMMSS}": begin_rel5_HHMMSS,
                    "{rel5_end_HHMMSS}": end_rel5_HHMMSS,
                    "{sim_direction}": rel5_sim_direction,  # sim_direction
                    "{rel5_min_1}": str(rel5_dom[0]),
                    "{rel5_min_2}": str(rel5_dom[1]),
                    "{rel5_max_1}": str(rel5_dom[2]),
                    "{rel5_max_2}": str(rel5_dom[3]),
                    "{rel5_ZPOINT_1}": rel5_ZPOINT_1,
                    "{rel5_ZPOINT_2}": rel5_ZPOINT_2,
                    "{rel5_domain_name}": rel5_dom_name,
                    "{rel5_ZTYPE}": rel5_ZTYPE,
                    "{rel5_NUMB_PART}": rel5_NUMB_PART,
                    "{rel5_XMASS}": rel5_XMASS}
    subdict_rel6 = {"{rel6_begin_YYYYMMDD}": begin_rel6_YYYYMMDD,
                    "{rel6_end_YYYYMMDD}": end_rel6_YYYYMMDD,
                    "{rel6_begin_HHMMSS}": begin_rel6_HHMMSS,
                    "{rel6_end_HHMMSS}": end_rel6_HHMMSS,
                    "{sim_direction}": rel6_sim_direction,  # sim_direction
                    "{rel6_min_1}": str(rel6_dom[0]),
                    "{rel6_min_2}": str(rel6_dom[1]),
                    "{rel6_max_1}": str(rel6_dom[2]),
                    "{rel6_max_2}": str(rel6_dom[3]),
                    "{rel6_ZPOINT_1}": rel6_ZPOINT_1,
                    "{rel6_ZPOINT_2}": rel6_ZPOINT_2,
                    "{rel6_domain_name}": rel6_dom_name,
                    "{rel6_ZTYPE}": rel6_ZTYPE,
                    "{rel6_NUMB_PART}": rel6_NUMB_PART,
                    "{rel6_XMASS}": rel6_XMASS}
    

    with open (file, 'r' ) as f:
        content = f.read()
        sub_relnum = "{rel_numb}"
        content_new = re.sub(sub_relnum, str(rel_numb), content, flags=re.M)
        content=content_new
        for key in subdict:
            print("00000000")
            print(key)
            print(subdict[key])
            content_new = re.sub(key, subdict[key], content, flags = re.M)
            content=content_new
        for key in subdict_rel2:
            print("00000000")
            print(key)
            print(subdict_rel2[key])
            content_new = re.sub(key, subdict_rel2[key], content, flags = re.M)
            content=content_new
            #if subdict_rel2[key] != "":
            #    rel_numb +=1
        for key in subdict_rel3:
            print("00000000")
            print(key)
            print(subdict_rel3[key])
            content_new = re.sub(key, subdict_rel3[key], content, flags = re.M)
            content=content_new
            #if subdict_rel3[key] != "":
            #    rel_numb += 1
        for key in subdict_rel4:
            print("00000000")
            print(key)
            print(subdict_rel4[key])
            content_new = re.sub(key, subdict_rel4[key], content, flags = re.M)
            content=content_new
        for key in subdict_rel5:
            print("00000000")
            print(key)
            print(subdict_rel5[key])
            content_new = re.sub(key, subdict_rel5[key], content, flags = re.M)
            content=content_new
        for key in subdict_rel6:
            print("00000000")
            print(key)
            print(subdict_rel6[key])
            content_new = re.sub(key, subdict_rel6[key], content, flags = re.M)
            content=content_new

        ff=open(file+dom_name,"w")
        ff.write(content_new.strip())
        ff

if __name__ == "__main__":
  import argparse
  def none_or_str(value):
    if value == 'None':
      return None
    return value
  parser = argparse.ArgumentParser()
  parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True,type=str)
  parser.add_argument("--steps", default=0, nargs="+", type=int,help="forecast times example --steps 0 3 gives time 0 to 3")
  parser.add_argument("--domain_name", default=None, help="see domain.py", type = none_or_str)
  parser.add_argument("--domain_lonlat", nargs="+", default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--point_name", default=None, help="see site.csv")
  parser.add_argument("--point_lonlat", nargs="+", type=float, default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--number_grid", type=int, default=500, help="")

  parser.add_argument("--begin_YYYYMMDD", type=str, default="", help="v")
  parser.add_argument("--end_YYYYMMDD", type=str,default="", help=" ")
  parser.add_argument("--begin_HHMMSS", type=str,default="", help=" ")
  parser.add_argument("--end_HHMMSS", type=str,default="", help=" ")
  parser.add_argument("--begin_rel_YYYYMMDD", type=str,default="", help=" ")
  parser.add_argument("--end_rel_YYYYMMDD",type=str, default="", help=" ")
  parser.add_argument("--begin_rel_HHMMSS",type=str, default="", help=" ")
  parser.add_argument("--end_rel_HHMMSS",type=str, default="", help=" ")

  parser.add_argument("--sim_direction",type=str, default="1", help="1:foreward, -1:backward")
  parser.add_argument("--file",type=str, default=None, help="path and filename")
  parser.add_argument("--ZPOINT_1",type=str, default="950", help="")
  parser.add_argument("--ZPOINT_2",type=str, default="800", help="")

  parser.add_argument("--begin_rel2_YYYYMMDD", type=str,default="", help=" ")
  parser.add_argument("--begin_rel2_HHMMSS",type=str, default="", help=" ")
  parser.add_argument("--end_rel2_YYYYMMDD", type=str,default="", help=" ")
  parser.add_argument("--end_rel2_HHMMSS",type=str, default="", help=" ")
  parser.add_argument("--begin_rel3_YYYYMMDD", type=str, default="", help=" ")
  parser.add_argument("--begin_rel3_HHMMSS", type=str, default="", help=" ")
  parser.add_argument("--end_rel3_YYYYMMDD", type=str, default="", help=" ")
  parser.add_argument("--end_rel3_HHMMSS", type=str, default="", help=" ")
  parser.add_argument("--begin_rel4_YYYYMMDD", type=str, default="", help=" ")
  parser.add_argument("--begin_rel4_HHMMSS", type=str, default="", help=" ")
  parser.add_argument("--end_rel4_YYYYMMDD", type=str, default="", help=" ")
  parser.add_argument("--end_rel4_HHMMSS", type=str, default="", help=" ")
  parser.add_argument("--begin_rel5_YYYYMMDD", type=str, default="", help=" ")
  parser.add_argument("--begin_rel5_HHMMSS", type=str, default="", help=" ")
  parser.add_argument("--end_rel5_YYYYMMDD", type=str, default="", help=" ")
  parser.add_argument("--end_rel5_HHMMSS", type=str, default="", help=" ")
  parser.add_argument("--begin_rel6_YYYYMMDD", type=str, default="", help=" ")
  parser.add_argument("--begin_rel6_HHMMSS", type=str, default="", help=" ")
  parser.add_argument("--end_rel6_YYYYMMDD", type=str, default="", help=" ")
  parser.add_argument("--end_rel6_HHMMSS", type=str, default="", help=" ")

  parser.add_argument("--rel2_ZPOINT_1", type=str, default="", help="")
  parser.add_argument("--rel2_ZPOINT_2", type=str, default="", help="")
  parser.add_argument("--rel3_ZPOINT_1", type=str, default="", help="")
  parser.add_argument("--rel3_ZPOINT_2", type=str, default="", help="")
  parser.add_argument("--rel4_ZPOINT_1", type=str, default="", help="")
  parser.add_argument("--rel4_ZPOINT_2", type=str, default="", help="")
  parser.add_argument("--rel5_ZPOINT_1", type=str, default="", help="")
  parser.add_argument("--rel5_ZPOINT_2", type=str, default="", help="")
  parser.add_argument("--rel6_ZPOINT_1", type=str, default="", help="")
  parser.add_argument("--rel6_ZPOINT_2", type=str, default="", help="")





  args = parser.parse_args()

  make_inputfile(datetime=args.datetime, steps = args.steps, domain_name = args.domain_name, domain_lonlat=args.domain_lonlat,
    point_name=args.point_name,point_lonlat=args.point_lonlat,number_grid=args.number_grid,
     begin_YYYYMMDD=args.begin_YYYYMMDD, end_YYYYMMDD=args.end_YYYYMMDD,
     begin_HHMMSS=args.begin_HHMMSS, end_HHMMSS=args.end_HHMMSS,
     begin_rel_YYYYMMDD=args.begin_rel_YYYYMMDD, end_rel_YYYYMMDD=args.end_rel_YYYYMMDD,
     begin_rel_HHMMSS=args.begin_rel_HHMMSS, end_rel_HHMMSS=args.end_rel_HHMMSS,
     sim_direction=args.sim_direction, file=args.file, ZPOINT_1 = args.ZPOINT_1, ZPOINT_2 = args.ZPOINT_2,
     begin_rel2_YYYYMMDD=args.begin_rel2_YYYYMMDD, begin_rel2_HHMMSS=args.begin_rel2_HHMMSS,
     end_rel2_YYYYMMDD=args.end_rel2_YYYYMMDD, end_rel2_HHMMSS=args.end_rel2_HHMMSS, 
     rel2_ZPOINT_1=args.rel2_ZPOINT_1, rel2_ZPOINT_2=args.rel2_ZPOINT_2,
     begin_rel3_YYYYMMDD=args.begin_rel3_YYYYMMDD, begin_rel3_HHMMSS=args.begin_rel3_HHMMSS,
     end_rel3_YYYYMMDD=args.end_rel3_YYYYMMDD, end_rel3_HHMMSS=args.end_rel3_HHMMSS,
     rel3_ZPOINT_1=args.rel3_ZPOINT_1,rel3_ZPOINT_2=args.rel3_ZPOINT_2,
     begin_rel4_YYYYMMDD=args.begin_rel4_YYYYMMDD, begin_rel4_HHMMSS=args.begin_rel4_HHMMSS,
     end_rel4_YYYYMMDD=args.end_rel4_YYYYMMDD, end_rel4_HHMMSS=args.end_rel4_HHMMSS,
     rel4_ZPOINT_1=args.rel4_ZPOINT_1,rel4_ZPOINT_2=args.rel4_ZPOINT_2,
     begin_rel5_YYYYMMDD=args.begin_rel5_YYYYMMDD, begin_rel5_HHMMSS=args.begin_rel5_HHMMSS,
     end_rel5_YYYYMMDD=args.end_rel5_YYYYMMDD, end_rel5_HHMMSS=args.end_rel5_HHMMSS,
     rel5_ZPOINT_1=args.rel5_ZPOINT_1, rel5_ZPOINT_2=args.rel5_ZPOINT_2,
     begin_rel6_YYYYMMDD=args.begin_rel6_YYYYMMDD, begin_rel6_HHMMSS=args.begin_rel6_HHMMSS,
     end_rel6_YYYYMMDD=args.end_rel6_YYYYMMDD, end_rel6_HHMMSS=args.end_rel6_HHMMSS,
     rel6_ZPOINT_1=args.rel6_ZPOINT_1, rel6_ZPOINT_2=args.rel6_ZPOINT_2,)

# python make_inputfile.py --domain_name Svalbard --datetime 2020030900 --begin_YYYYMMDD 20200309 --end_YYYYMMDD 20200311 --begin_HHMMSS 000000 --end_HHMMSS 000000 --begin_rel_YYYYMMDD 20200309 --end_rel_YYYYMMDD 20200309 --begin_rel_HHMMSS 000000 --end_rel_HHMMSS 000000

#python make_inputfile.py --domain_name Svalbard --datetime 2021032506 --begin_YYYYMMDD 20210325 --end_YYYYMMDD 20200311 --begin_HHMMSS 000000 --end_HHMMSS 000000 --file flexarome.input-ISLAS-lowres__INPUT_multrel
# --begin_rel_YYYYMMDD 20200309 --end_rel_YYYYMMDD 20200309 --begin_rel_HHMMSS 000000 --end_rel_HHMMSS 000000 --begin_rel2_YYYYMMDD 20200309 --rel4_begin_HHMMSS 000000
