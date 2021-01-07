# %%
# python Z500_VEL.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway

from weathervis.config import *
from weathervis.check_data import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.calculation import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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

def Z500_VEL(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None, legend=False, info = False):

  for dt in datetime: #modelrun at time..
    date = dt[0:-2]
    hour = int(dt[-2:])
    param_sfc = ["air_temperature_2m", "air_pressure_at_sea_level", "surface_geopotential"]
    param_pl = ["air_temperature_pl", "y_wind_pl", "x_wind_pl", "geopotential_pl"]
    param_sfx = ["SST","SIC"] #add later
    p_levels = [850,1000]
    param = param_sfc + param_pl
    split = False
    print("\n######## Checking if your request is possibel ############")
    try:
      check_all = check_data(date=dt, model=model, param=param, levtype="pl", p_level=p_levels, step=steps)
      check_sfx = check_data(date=dt, model=model, param=param_sfx, step=steps)

    except ValueError:
      split = True
      try:
        print("--------> Splitting up your request to find match ############")
        check_sfc = check_data(date=dt, model=model, param=param_sfc)
        check_pl = check_data(date=dt, model=model, param=param_pl, levtype="pl", p_level=p_levels)
        check_sfx = check_data(date=dt, model=model, param=param_sfx)

        print(check_pl.file)
      except ValueError:
        print("!!!!! Sorry this plot is not availbale for this date. Try with another datetime !!!!!")
        break
    print("--------> Found match for your request ############")


    if not split:
      file_all = check_all.file.loc[0]

      data_domain = domain_input_handler(dt, model,domain_name, domain_lonlat, file_all)

      #lonlat = np.array(data_domain.lonlat)
      dmap_meps = get_data(model=model, data_domain=data_domain, param=param, file=file_all, step=steps,
                           date=dt, p_level=p_levels)
      dmap_mepsdfx = get_data(model=model, data_domain=data_domain, param=param_sfx, file=check_sfx.file.loc[0], step=steps,date=dt)
      print("\n######## Retriving data ############")
      print(f"--------> from: {dmap_meps.url} ")
      dmap_meps.retrieve()
      tmap_meps = dmap_meps # two names for same value, no copying done.
      dmap_mepsdfx.retrieve()
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
      file_pl = check_pl.file.loc[0]
      tmap_meps = get_data(model=model, data_domain=data_domain, param=param_pl, file=file_pl, step=steps, date=dt, p_level=p_levels)
      print("\n######## Retriving data ############")
      print(f"--------> from: {tmap_meps.url} ")
      tmap_meps.retrieve()

      dmap_mepsdfx = get_data(model=model, data_domain=data_domain, param=param_sfx, file=check_sfx.file.loc[0],
                              step=steps,
                              date=dt)
      dmap_mepsdfx.retrieve()

    #CALCULATE
    u, v = xwind2uwind(tmap_meps.x_wind_pl, tmap_meps.y_wind_pl, tmap_meps.alpha)
    vel = wind_speed(tmap_meps.x_wind_pl, tmap_meps.y_wind_pl)

    pt = potential_temperatur(dmap_meps.air_temperature_pl, dmap_meps.pressure*100.)
    pt_sst = potential_temperatur(dmap_mepsdfx.SST, dmap_meps.air_pressure_at_sea_level[:,0,:,:])

    dpt = pt[:,np.where(dmap_meps.pressure==1000)[0],:,:]-pt[:,np.where(dmap_meps.pressure==850)[0],:,:]
    dpt_sst =pt_sst[:,:,:] - pt[:,np.where(dmap_meps.pressure==850)[0],:,:].squeeze()
    #dpt_sst =abs(pt_sst[:,:,:] - pt[:,np.where(dmap_meps.pressure==850)[0],:,:].squeeze())

    # convert fields
    dmap_meps.air_pressure_at_sea_level/=100
    tmap_meps.geopotential_pl/=10.0

    # plot map
    fig1 = plt.figure(figsize=(7, 9))

    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    parallels = dmap_meps.standard_parallel_projection_lambert

    # setting up projection
    # setting up projection
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
    crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                 globe=globe)

    for tim in np.arange(np.min(steps), np.max(steps)+1, 1):
      ax1 = plt.subplot(projection=crs)

      ttt = tim #+ np.min(steps)
      tidx = tim - np.min(steps)
      print('Plotting {0} + {1:02d} UTC'.format(dt, ttt))
      plev2 = 0
      embr = 0
      ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
      MSLP = np.where(ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()
      VEL = (vel[tidx, plev2, :, :]).squeeze()
      Z = (tmap_meps.geopotential_pl[tidx, plev2, :, :]).squeeze()
      DELTAPT=dpt[tidx, 0, :, :]
      DELTAPT = dpt_sst[tidx,:,:]
      ICE = dmap_mepsdfx.SIC[tidx, :, :]
      DELTAPT = np.where( ICE <= 0.99,DELTAPT,0)
      lvl = range(-1,13)
      C = [[255,255,255	],  # grey #[255,255,255],#gre
           [204,191,189	],  # grey
           [155,132,127	],  # grey
           [118,86,80],  # lillac, 39	64	197	149,53,229
           [138,109,81],  # blue dark,7,67,194 [218,81,14],
           [181,165,102],  #
           [229,226,124],  ##
           [213,250,128],
           [125,231,111],
           [55,212,95],
           [25,184,111],
           [17,138,234],
           [21,82,198],
           [37,34,137]]
      C = np.array(C)
      C = np.divide(C, 255.)  # RGB has to be between 0 and 1 in python
      CF_prec = plt.contourf(dmap_meps.x, dmap_meps.y, DELTAPT, zorder=0,
                            antialiased=True,extend = "max", levels=lvl, colors=C, vmin=0, vmax=12)#

      CF_ice = plt.contour(dmap_meps.x, dmap_meps.y, ICE, zorder=1, linewidths=5, colors="black", levels=[0.1, 0.8, 0.99])  #
      # MSLP with contour labels every 10 hPa
      C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=1, alpha=1.0,
                      levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 1),
                      colors='grey', linewidths=0.5)
      C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=2, alpha=1.0,
                        levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 10),
                        colors='grey', linewidths=1.0)
      ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

      #CS = ax1.contour(dmap_meps.x, dmap_meps.y, Z, zorder=3, alpha=1.0,
      #                  levels=np.arange(4600, 5800, 20), colors="blue", linewidths=0.7)
      #ax1.clabel(CS, CS.levels, inline=True, fmt="%4.0f", fontsize=10)

      ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'))  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).

      ##########################################################
      legend=True
      if legend:
        proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="grey"),
                plt.axhline(y=0, xmin=1, xmax=1, color="black",linewidth=4)]
        try:
          plt.colorbar(CF_prec, fraction=0.046, pad=0.01, aspect=25, label=r"$\theta_{SST}-\theta_{850}$", extend="both")

        except:
          pass

        lg = ax1.legend(proxy, [f"MSLP [hPa]",
                               f"Sea ice at 10%, 80%, 99%"])
        frame = lg.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(1)
      fig1.savefig("../../../output/{0}_CAOi_{1}+{2:02d}.png".format(model, dt, ttt), bbox_inches="tight", dpi=200)
      ax1.cla()
      plt.clf()
# fin

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
  Z500_VEL(datetime=args.datetime, steps = args.steps, model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info)
  #datetime, step=4, model= "MEPS", domain = None
