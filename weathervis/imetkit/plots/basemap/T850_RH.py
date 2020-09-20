
#python T850_RH.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway

from imetkit.domain import *  # require netcdf4
from imetkit.check_data import *
from imetkit.get_data import *
from imetkit.calculation import *
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import warnings

# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)
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

def T850_RH(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None, legend=False, info = False):
  for dt in datetime: #modelrun at time..
    date = dt[0:-2]
    hour = int(dt[-2:])
    param_sfc = ["air_pressure_at_sea_level", "air_temperature_2m", "precipitation_amount_acc", "surface_geopotential"]
    param_pl = ["air_temperature_pl", "relative_humidity_pl"]
    param = param_sfc + param_pl
    #print(type(steps))
    split = False
    print("\n######## Checking if your request is possibel ############")
    try:
      check_all = check_data(date=dt, model=model, param=param, p_level=850)
    except ValueError:
      split = True
      try:
        print("--------> Splitting up your request to find match ############")
        check_sfc = check_data(date=dt, model=model, param=param_sfc)
        check_pl = check_data(date=dt, model=model, param=param_pl, p_level=850)
      except ValueError:
        print("!!!!! Sorry this plot is not availbale for this date. Try with another datetime !!!!!")
        break
    print("--------> Found match for your request ############")

    if not split:
      file_all = check_all.file.loc[0]

      data_domain = domain_input_handler(dt, model,domain_name, domain_lonlat, file_all)

      #lonlat = np.array(data_domain.lonlat)
      dmap_meps = get_data(model=model, data_domain=data_domain, param=param, file=file_all, step=steps,
                           date=dt, p_level=850)
      print("\n######## Retriving data ############")
      print(f"--------> from: {dmap_meps.url} ")
      dmap_meps.retrieve()
      tmap_meps = dmap_meps # two names for same value, no copying done.
    else:
      # get sfc level data
      file_sfc = check_sfc.file.loc[0]
      data_domain = domain_input_handler(dt, model,domain_name, domain_lonlat, file_sfc)
      #lonlat = np.array(data_domain.lonlat)
      dmap_meps = get_data(model=model, param=param_sfc, file=file_sfc, step=steps, date=dt, data_domain=data_domain)
      print("\n######## Retriving data ############")
      print(f"--------> from: {dmap_meps.url} ")
      dmap_meps.retrieve()

      # get pressure level data
      file_pl = check_pl.file
      tmap_meps = get_data(model=model, data_domain=data_domain, param=param_pl, file=file_pl, step=steps, date=dt,  p_level=850)
      print("\n######## Retriving data ############")
      print(f"--------> from: {tmap_meps.url} ")
      tmap_meps.retrieve()

    # convert fields
    dmap_meps.air_pressure_at_sea_level /= 100
    dmap_meps.air_temperature_2m -= 273.15
    tmap_meps.air_temperature_pl -= 273.15
    tmap_meps.relative_humidity_pl *= 100.0

    # plot map
    fig1, ax1 = plt.subplots(figsize=(7, 9))
    lonlat = [dmap_meps.longitude[0,0], dmap_meps.longitude[-1,-1], dmap_meps.latitude[0,0], dmap_meps.latitude[-1,-1]]
    print(lonlat)

    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    map = Basemap(llcrnrlon=lonlat[0], llcrnrlat=lonlat[2], urcrnrlon=lonlat[1], urcrnrlat=lonlat[3],
                  resolution='i', projection="lcc", lon_0=lon0, lat_0=lat0, lat_1=lat0, area_thresh=0.0001)
    x, y = map(dmap_meps.longitude, dmap_meps.latitude) #longitude_of_central_meridian

    for tim in np.arange(np.min(steps), np.max(steps)+1,1):
      tidx = tim - np.min(steps)
      print('Plotting {0} + {1:02d} UTC'.format(dt,tim))
      # gather, filter and squeeze variables for plotting
      plev = 0
      #reduces noise over mountains by removing values over a certain height.
      Z = dmap_meps.surface_geopotential[tidx, 0, :, :]
      TA = np.where(Z < 3000, tmap_meps.air_temperature_pl[tidx, plev, :, :], np.NaN).squeeze()
      MSLP = np.where(Z < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()
      RH = (tmap_meps.relative_humidity_pl[tidx, plev, :, :]).squeeze()


      # clear subplot
      plt.cla()
      # MSLP with contour labels every 10 hPa
      C_P = plt.contour(x, y, MSLP, zorder=1, alpha=1.0,
                        levels=np.arange(round(np.nanmin(MSLP),-1)-10, round(np.nanmax(MSLP),-1)+10, 1), colors='grey', linewidths=0.5)
      C_P = plt.contour(x, y, MSLP, zorder=2, alpha=1.0,
                        levels=np.arange(round(np.nanmin(MSLP),-1)-10, round(np.nanmax(MSLP),-1)+10, 10),
                       colors='grey', linewidths=1.0, label = "MSLP [hPa]")
      ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)
      # air temperature (C)
      C_T = plt.contour(x, y, TA, zorder=3, alpha=1.0,
                         levels=np.arange(-50, 30, 0.5), colors="red", linewidths=0.7, label = "Temp [C]")
      # relative humidity above 80%
      CF_RH = plt.contourf(x, y, RH, zorder=1, alpha=0.1,
                        levels=np.arange(80, 120, 20), colors="blue", linewidths=0.7,label = "RH >80% [%]")
      map.drawcoastlines(linewidth=0.5, color='black', ax=ax1, zorder=1)


      proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0], )
               for pc in CF_RH.collections]
      proxy1 = [plt.axhline(y=0, xmin=1, xmax=1, color="red"),
                plt.axhline(y=0, xmin=1, xmax=1, color="red", linestyle="dashed"),
                plt.axhline(y=0, xmin=1, xmax=1, color="gray")]
      proxy.extend(proxy1)

      if legend:
        lg = ax1.legend(proxy, [f"RH > 80% [%] at {dmap_meps.pressure[plev]:.0f} hPa",
                              f"T>0 [C] at {dmap_meps.pressure[plev]:.0f} hPa",
                              f"T<0 [C] at {dmap_meps.pressure[plev]:.0f} hPa", "MSLP [hPa]", ""])
        frame = lg.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(1)

      if info:
        plt.text(x=0, y=-1, s="INFO: Reduced topographic noise by filtering with surface_geopotential bellow 3000",
                 fontsize=7)#, bbox=dict(facecolor='white', alpha=0.5))

      fig2 = plt.figure(figsize=(2, 1.25))
      fig2.legend(proxy, [f"RH > 80% [%] at {dmap_meps.pressure[plev]:.0f} hPa",
                              f"T>0 [C] at {dmap_meps.pressure[plev]:.0f} hPa",
                              f"T<0 [C] at {dmap_meps.pressure[plev]:.0f} hPa", "MSLP [hPa]", ""])
      fig2.savefig("../../../output/{0}_T850_RH_LEGEND.png".format(model),bbox_inches="tight", dpi=200)
      ##########################################################

      #plt.show()
      fig1.savefig("../../../output/{0}_T850_RH_{1}+{2:02d}.png".format(model,dt, tim),bbox_inches="tight", dpi=200)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
  parser.add_argument("--steps", default=0, nargs="+", type=int,help="forecast times example --steps 0 3 gives time 0 to 3")
  parser.add_argument("--model",default="MEPS", help="MEPS or AromeArctic")
  parser.add_argument("--domain_name", default=None, help="MEPS or AromeArctic")
  parser.add_argument("--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--legend", default=False, help="Display legend")
  parser.add_argument("--info", default=False, help="Display info")
  args = parser.parse_args()
  T850_RH(datetime=args.datetime, steps = args.steps, model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info)
  #datetime, step=4, model= "MEPS", domain = None


