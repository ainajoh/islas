# %%
#python T850_RH.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway
#
from weathervis.config import *
from weathervis.utils import *

#import weathervis.config as wc
import cartopy.crs as ccrs
from weathervis.domain import *  # require netcdf4
from weathervis.check_data import *
from weathervis.get_data import *
from weathervis.calculation import *
import matplotlib.pyplot as plt
import warnings
import cartopy.feature as cfeature
from add_overlays import *

print("done")
# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=Downloading)

def T850_RH(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None, legend=False, info = False, save = True, grid=True, runid=None, outpath=None):
  global OUTPUTPATH
  if outpath != None:
      OUTPUTPATH=outpath

  for dt in datetime: #modelrun at time..
    if runid !=None:
        make_modelrun_folder = setup_directory( OUTPUTPATH, "{0}-{1}".format(dt,runid) )
    else:
        make_modelrun_folder = setup_directory( OUTPUTPATH, "{0}".format(dt) )
    
    print(dt)
    date = dt[0:-2]
    hour = int(dt[-2:])
    param_sfc = ["air_pressure_at_sea_level", "air_temperature_2m", "surface_geopotential"]
    param_pl = ["air_temperature_pl", "relative_humidity_pl"]
    param = param_sfc + param_pl
    #print(type(steps))
    split = False
    print("\n######## Checking if your request is possible ############")
    try:
      check_all = check_data(date=dt, model=model, param=param, p_level = 850, step=steps)
    except ValueError:
      split = True
      try:
        print("--------> Splitting up your request to find match ############")
        check_sfc = check_data(date=dt, model=model, param=param_sfc,step=steps)
        check_pl = check_data(date=dt, model=model, param=param_pl, p_level=850,step=steps)
      except ValueError:
        print("!!!!! Sorry this plot is not available for this date. Try with another datetime !!!!!")
        break
    print("--------> Found match for your request ############")

    if not split:
      file_all = check_all.file.loc[0]

      data_domain = domain_input_handler(dt, model,domain_name, domain_lonlat, file_all)

      #lonlat = np.array(data_domain.lonlat)
      dmap_meps = get_data(model=model, data_domain=data_domain, param=param, file=file_all, step=steps,
                           date=dt, p_level=[850])
      print("\n######## Retrieving data ############")
      print(f"--------> from: {dmap_meps.url} ")
      dmap_meps.retrieve()
      tmap_meps = dmap_meps # two names for same value, no copying done.
    else:
      # get sfc level data
      file_sfc = check_sfc.file.loc[0]
      data_domain = domain_input_handler(dt, model,domain_name, domain_lonlat, file_sfc)
      #lonlat = np.array(data_domain.lonlat)
      dmap_meps = get_data(model=model, param=param_sfc, file=file_sfc, step=steps, date=dt, data_domain=data_domain)
      print("\n######## Retrieving data ############")
      print(f"--------> from: {dmap_meps.url} ")
      dmap_meps.retrieve()

      # get pressure level data
      file_pl = check_pl.file
      tmap_meps = get_data(model=model, data_domain=data_domain, param=param_pl, file=file_pl, step=steps, date=dt, p_level = 850)
      print("\n######## Retrieving data ############")
      print(f"--------> from: {tmap_meps.url} ")
      tmap_meps.retrieve()

    # convert fields
    dmap_meps.air_pressure_at_sea_level /= 100
    dmap_meps.air_temperature_2m -= 273.15
    tmap_meps.air_temperature_pl -= 273.15
    tmap_meps.relative_humidity_pl *= 100.0

    # plot map


    lonlat = [dmap_meps.longitude[0,0], dmap_meps.longitude[-1,-1], dmap_meps.latitude[0,0], dmap_meps.latitude[-1,-1]]
    print(lonlat)

    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    parallels = dmap_meps.standard_parallel_projection_lambert


    # setting up projection
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
    crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                globe=globe)
    #fig1 = plt.figure(figsize=(7, 9))
                               
    for tim in np.arange(np.min(steps), np.max(steps)+1,1):
      #ax1 = plt.subplot(projection=crs)
    
      # determine if image should be created for this time step
      stepok=False
      if tim<25:
          stepok=True
      elif (tim<=36) and ((tim % 3) == 0):
          stepok=True
      elif (tim<=66) and ((tim % 6) == 0):
          stepok=True
      if stepok==True:

          fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9),subplot_kw={'projection': crs})
          ttt = tim
          tidx = tim - np.min(steps)

          print('Plotting T850 {0} + {1:02d} UTC'.format(dt,tim))
          # gather, filter and squeeze variables for plotting
          plev = 0
          #reduces noise over mountains by removing values over a certain height.

          Z = dmap_meps.surface_geopotential[tidx, 0, :, :]
          TA = np.where(Z < 3000, tmap_meps.air_temperature_pl[tidx, plev, :, :], np.NaN).squeeze()
          MSLP = np.where(Z < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()
          RH = (tmap_meps.relative_humidity_pl[tidx, plev, :, :]).squeeze()
          # MSLP with contour labels every 10 hPa
          C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=2, alpha=1.0,
                            levels=np.arange(960, 1050, 1), colors='grey', linewidths=0.5)
          C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=3, alpha=1.0,
                            levels=np.arange(960, 1050, 10),
                           colors='grey', linewidths=1.0, label = "MSLP [hPa]")
          ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)
          # air temperature (C)
          CF_T= ax1.contourf(dmap_meps.x, dmap_meps.y, TA, zorder=1, alpha=1.0,
                             levels=np.arange(-40, 20, 1.0), cmap="PRGn")
          C_T = ax1.contour(dmap_meps.x, dmap_meps.y, TA, zorder=4, alpha=1.0,
                              levels=np.arange(-40, 20, 1.0), colors="red", linewidths=0.7)
          ax1.clabel(C_T, C_T.levels[::3], inline=True, fmt="%3.0f", fontsize=10)

          # relative humidity above 80%
          CF_RH = ax1.contour(dmap_meps.x, dmap_meps.y, RH, zorder=4, alpha=0.5,
                            levels=np.linspace(70, 100, 4), colors="blue", linewidths=0.7,label = "RH >70% [%]")

          #lat_p = 60.2
          #lon_p = 5.4167
          #mainpoint = ax1.scatter(lon_p, lat_p, s=9.0 ** 2, transform=ccrs.PlateCarree(),
          #                        color='lime', zorder=6, linestyle='None', edgecolors="k", linewidths=3)

          ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'))  #‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
          ax1.text(0, 1, "{0}_T850_RH_{1}+{2:02d}".format(model, dt, ttt), ha='left', va='bottom', transform=ax1.transAxes, color='black')

          legend=False
          if legend:
            proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0], )
                    for pc in CF_T.collections]
            proxy1 = [plt.axhline(y=0, xmin=1, xmax=1, color="red"),
                     plt.axhline(y=0, xmin=1, xmax=1, color="red", linestyle="dashed"),
                     plt.axhline(y=0, xmin=1, xmax=1, color="gray")]
            proxy.extend(proxy1)
            lg = ax1.legend(proxy, [f"RH > 80% [%] at {dmap_meps.pressure[plev]:.0f} hPa",
                                  f"T>0 [C] at {dmap_meps.pressure[plev]:.0f} hPa",
                                  f"T<0 [C] at {dmap_meps.pressure[plev]:.0f} hPa", "MSLP [hPa]", ""])
            frame = lg.get_frame()
            frame.set_facecolor('white')
            frame.set_alpha(1)

          #if info:
          #  plt.text(x=0, y=-1, s="INFO: Reduced topographic noise by filtering with surface_geopotential bellow 3000",
          #           fontsize=7)#, bbox=dict(facecolor='white', alpha=0.5))


          ##########################################################

          #plt.show()


          #lonlat = [dmap_meps.longitude[0, 0], dmap_meps.longitude[-1, -1], dmap_meps.latitude[0, 0],
          #          dmap_meps.latitude[-1, -1]]
          # ax.set_extent((lonlat[0]-5, lonlat[1], lonlat[2], lonlat[3]))  # (x0, x1, y0, y1)
          # ax.set_extent((dmap_meps.x[0], dmap_meps.x[-1], dmap_meps.y[0], dmap_meps.y[-1]))  # (x0, x1, y0, y1)
          #ax1.set_extent((lonlat[0], lonlat[1], lonlat[2], lonlat[3]))
          #fig1.savefig("../../../../output/{0}_T850_RH_{1}_{2:02d}.png".format(model,dt, tim), bbox_inches="tight", dpi=200)

          if grid:
            nicegrid(ax=ax1)

          add_ISLAS_overlays(ax1,col='black')

          if domain_name != model and data_domain != None:  # weird bug.. cuts off when sees no data value
            ax1.set_extent(data_domain.lonlat)

          print(make_modelrun_folder+"/{0}_{1}_T850_RH_{2}+{3:02d}.png".format(model, domain_name, dt, tim))
          fig1.savefig(make_modelrun_folder+"/{0}_{1}_T850_RH_{2}+{3:02d}.png".format(model, domain_name, dt, tim), bbox_inches="tight", dpi=200)
          ax1.cla()
          plt.clf()
          plt.close(fig1)

    #proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0], )
    #        for pc in CF_RH.collections]
    #proxy1 = [plt.axhline(y=0, xmin=1, xmax=1, color="red"),eee
    #         plt.axhline(y=0, xmin=1, xmax=1, color="red", linestyle="dashed"),
    #         plt.axhline(y=0, xmin=1, xmax=1, color="gray")]
    #proxy.extend(proxy1)
    #fig2 = plt.figure(figsize=(2, 1.25))
    #fig2.legend(proxy, [f"RH > 80% [%] at {dmap_meps.pressure[plev]:.0f} hPa",
    #                 f"T>0 [C] at {dmap_meps.pressure[plev]:.0f} hPa",
    #                  f"T<0 [C] at {dmap_meps.pressure[plev]:.0f} hPa", "MSLP [hPa]", ""])
    #fig2.savefig("../../../output/{0}_T850_RH_LEGEND.png".format(model), bbox_inches="tight", dpi=200)

  plt.close("all")


if __name__ == "__main__":
  import argparse

  def none_or_str(value):
    if value == 'None':
      return None
    return value

  parser = argparse.ArgumentParser()
  parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
  parser.add_argument("--steps", default=0, nargs="+", type=int,help="forecast times example --steps 0 3 gives time 0 to 3")
  parser.add_argument("--model",default="MEPS", help="MEPS or AromeArctic")
  parser.add_argument("--domain_name", default=None, help="see domain.py", type = none_or_str)
  parser.add_argument("--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--legend", default=False, help="Display legend")
  parser.add_argument("--grid", default=True, help="Display legend")
  parser.add_argument("--id", default=None, help="Display legend", type=str)
  parser.add_argument("--outpath", default=None, help="Display legend", type=str)
  parser.add_argument("--info", default=False, help="Display info")
  args = parser.parse_args()
  print(args.__dict__)
  
  T850_RH(datetime=args.datetime, steps = args.steps, model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info, grid=args.grid,  runid =args.id, outpath=args.outpath)

