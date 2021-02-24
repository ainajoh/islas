from weathervis.config import *
from weathervis.utils import *
from weathervis.check_data import *
from weathervis.domain import *
from weathervis.calculation import *
from weathervis.get_data import *

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

def point(datetime, point_name, point_lonlat,number_grid=500):
    if point_name:
        sites = pd.read_csv("../../weathervis/weathervis/data/sites.csv", sep=";", header=0, index_col=0)
        point_lonlat = [sites.loc[point_name].lon, sites.loc[point_name].lat]

    #test nearest lonlat
    check_all = check_data(date=datetime, model="AromeArctic", param=["longitude"], step=1)
    file_all = check_all.file
    data_domain = domain_input_handler(dt=datetime, model="AromeArctic", domain_name="AromeArctic", file=file_all,domain_lonlat=None)
    dmap_meps = get_data(model="AromeArctic", data_domain=data_domain, param=["longitude"], file=file_all, step=1,
                         date=datetime)
    dmap_meps.retrieve()
    closest_idx= nearest_neighbour_idx(point_lonlat[0], point_lonlat[1], dmap_meps.longitude, dmap_meps.latitude, nmin=number_grid)

    x_llc=np.min(closest_idx[0])
    y_llc = np.min(closest_idx[1])
    x_urc = np.max(closest_idx[0])
    y_urc = np.max(closest_idx[1])
    #print(x_llc*2500)
    #print(y_llc * 2500)
    #print(x_urc * 2500)
    #print(y_urc * 2500)
    return [x_llc*2500,y_llc * 2500, x_urc * 2500, y_urc * 2500]
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
def make_inputfile(datetime, steps, domain_name, domain_lonlat, point_name, point_lonlat, number_grid,
     begin_YYYYMMDD, end_YYYYMMDD,
     begin_HHMMSS, end_HHMMSS,
     begin_rel_YYYYMMDD, end_rel_YYYYMMDD,
     begin_rel_HHMMSS, end_rel_HHMMSS,
     sim_direction,file):

    dom_name="lonlat"
    if point_name or point_lonlat:
        dom = point(datetime, point_name, point_lonlat, number_grid)
        if point_name:
            dom_name= point_name
    elif domain_name or domain_lonlat:
        dom = area(datetime, domain_name, domain_lonlat)
        if domain_name:
            dom_name= point_name
    #file="/Users/ainajoh/flexarome.input-ISLAS-lowres__INPUT"
    import re
    subdict={"{date_input_for_flexpart}":datetime,
             "{begin_YYYYMMDD}":begin_YYYYMMDD,
             "{end_YYYYMMDD}":end_YYYYMMDD,
             "{begin_HHMMSS}":begin_HHMMSS,
             "{end_HHMMSS}":end_HHMMSS,
             "{begin_rel_YYYYMMDD}":begin_rel_YYYYMMDD,
             "{end_rel_YYYYMMDD}":end_rel_YYYYMMDD,
             "{begin_rel_HHMMSS}":begin_rel_HHMMSS,
             "{end_rel_HHMMSS}":end_rel_HHMMSS,
             "{sim_direction}":sim_direction,#sim_direction
             "{min_1}":str(dom[0]),
             "{min_2}":str(dom[1]),
             "{max_1}":str(dom[2]),
             "{max_2}":str(dom[3]),
             "{domain_name}":dom_name}

    with open (file, 'r' ) as f:
        content = f.read()
        for key in subdict:
            print("00000000")
            print(key)
            print(subdict[key])
            content_new = re.sub(key, subdict[key], content, flags = re.M)
            content=content_new



        ff=open(file+dom_name,"w")
        ff.write(content_new)

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
  parser.add_argument("--number_grid", type=int, default=None, help="")

  parser.add_argument("--begin_YYYYMMDD", type=str, default=None, help="v")
  parser.add_argument("--end_YYYYMMDD", type=str,default=None, help=" ")
  parser.add_argument("--begin_HHMMSS", type=str,default=None, help=" ")
  parser.add_argument("--end_HHMMSS", type=str,default=None, help=" ")
  parser.add_argument("--begin_rel_YYYYMMDD", type=str,default=None, help=" ")
  parser.add_argument("--end_rel_YYYYMMDD",type=str, default=None, help=" ")
  parser.add_argument("--begin_rel_HHMMSS",type=str, default=None, help=" ")
  parser.add_argument("--end_rel_HHMMSS",type=str, default=None, help=" ")

  parser.add_argument("--sim_direction",type=str, default="1", help="1:foreward, -1:backward")
  parser.add_argument("--file",type=str, default=None, help="path and filename")



  args = parser.parse_args()

  make_inputfile(datetime=args.datetime, steps = args.steps, domain_name = args.domain_name, domain_lonlat=args.domain_lonlat,
    point_name=args.point_name,point_lonlat=args.point_lonlat,number_grid=args.number_grid,
     begin_YYYYMMDD=args.begin_YYYYMMDD, end_YYYYMMDD=args.end_YYYYMMDD,
     begin_HHMMSS=args.begin_HHMMSS, end_HHMMSS=args.end_HHMMSS,
     begin_rel_YYYYMMDD=args.begin_rel_YYYYMMDD, end_rel_YYYYMMDD=args.end_rel_YYYYMMDD,
     begin_rel_HHMMSS=args.begin_rel_HHMMSS, end_rel_HHMMSS=args.end_rel_HHMMSS,
     sim_direction=args.sim_direction, file=args.file)
  #datetime, step=4, model= "MEPS", domain = None


# python make_inputfile.py --domain_name Svalbard --datetime 2020030900 --begin_YYYYMMDD 20200309 --end_YYYYMMDD 20200311 --begin_HHMMSS 000000 --end_HHMMSS 000000 --begin_rel_YYYYMMDD 20200309 --end_rel_YYYYMMDD 20200309 --begin_rel_HHMMSS 000000 --end_rel_HHMMSS 000000
