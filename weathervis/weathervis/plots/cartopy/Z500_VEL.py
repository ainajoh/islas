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
    param_sfc = ["air_pressure_at_sea_level", "precipitation_amount_acc", "surface_geopotential"]
    param_pl = ["x_wind_pl", "y_wind_pl", "geopotential_pl"]
    param = param_sfc + param_pl
    plevel = [500]
    split = False
    print("\n######## Checking if your request is possibel ############")
    try:
      check_all = check_data(date=dt, model=model, param=param, levtype="pl", p_level=plevel, step=steps)
      print(check_all.file)

    except ValueError:
      split = True
      try:
        print("--------> Splitting up your request to find match ############")
        check_sfc = check_data(date=dt, model=model, param=param_sfc, step=steps)
        check_pl = check_data(date=dt, model=model, param=param_pl, levtype="pl", p_level=plevel, step=steps)
        print(check_pl.file)
      except ValueError:
        print("!!!!! Sorry this plot is not availbale for this date. Try with another datetime !!!!!")
        break
    print("--------> Found match for your request ############")


    if not split:
      file_all = check_all.file.loc[0]

      data_domain = domain_input_handler(dt, model,domain_name, domain_lonlat, file_all)

      #lonlat = np.array(data_domain.lonlat)
      print(file_all)
      dmap_meps = get_data(model=model, data_domain=data_domain, param=param, file=file_all, step=steps,
                           date=dt, p_level=plevel)
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
      file_pl = check_pl.file.loc[0]
      tmap_meps = get_data(model=model, data_domain=data_domain, param=param_pl, file=file_pl, step=steps, date=dt, p_level=plevel)
      print("\n######## Retriving data ############")
      print(f"--------> from: {tmap_meps.url} ")
      tmap_meps.retrieve()


    # convert fields
    dmap_meps.air_pressure_at_sea_level/=100
    #dmap_meps.precipitation_amount_acc*=1000.0
    print(dmap_meps.units.precipitation_amount_acc)
    tmap_meps.geopotential_pl/=10.0
    u,v = xwind2uwind(tmap_meps.x_wind_pl,tmap_meps.y_wind_pl, tmap_meps.alpha)
    vel = wind_speed(tmap_meps.x_wind_pl,tmap_meps.y_wind_pl)

    # plot map
    fig1 = plt.figure(figsize=(7, 9))

    lonlat = [dmap_meps.longitude[0, 0], dmap_meps.longitude[-1, -1], dmap_meps.latitude[0, 0],
              dmap_meps.latitude[-1, -1]]

    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    parallels = dmap_meps.standard_parallel_projection_lambert

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
      TP = precip_acc(dmap_meps.precipitation_amount_acc, acc=1)[tidx, 0, :,:].squeeze()
      VEL = (vel[tidx, plev2, :, :]).squeeze()
      Z = (tmap_meps.geopotential_pl[tidx, plev2, :, :]).squeeze()


      cmap = plt.get_cmap("viridis")# PuBuGn PuBuGn, nipy_spectral twilight  , plasma, gist_ncar viridis  inferno ,,,rainbow
      #lvl = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
      lvl = [ 0.01, 0.2, 0.5, 1, 3, 5, 10, 15, 20, 25, 30]
      norm = mcolors.BoundaryNorm(lvl, cmap.N)

      try: #workaround for a stupid matplotlib error not handling when all values are outside of range in lvl or all just nans..
        #https://github.com/SciTools/cartopy/issues/1290
        #cmap =  mcolors.ListedColormap('hsv', 'hsv') #plt.get_cmap("hsv")PuBu
        #TP.filled(np.nan) #fill mask with nan to avoid:  UserWarning: Warning: converting a masked element to nan.
        CF_prec = plt.contourf(dmap_meps.x, dmap_meps.y, TP, zorder=1,
                               cmap=cmap, norm = norm, alpha=0.6, antialiased=True,
                               levels=lvl, extend = "max")#
      except:
        pass
      # MSLP with contour labels every 10 hPa
      C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=1, alpha=1.0,
                      levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 1),
                      colors='grey', linewidths=0.5)
      C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=2, alpha=1.0,
                        levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 10),
                        colors='grey', linewidths=1.0)
      ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

      CS = ax1.contour(dmap_meps.x, dmap_meps.y, VEL, zorder=3, alpha=1.0,
                         levels=np.arange(-80, 80, 5), colors="green", linewidths=0.7)
      # geopotential
      CS = ax1.contour(dmap_meps.x, dmap_meps.y, Z, zorder=3, alpha=1.0,
                        levels=np.arange(4600, 5800, 20), colors="blue", linewidths=0.7)
      ax1.clabel(CS, CS.levels, inline=True, fmt="%4.0f", fontsize=10)

      ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'))  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).

      ##########################################################

      if legend:
        proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="green"),
        plt.axhline(y=0, xmin=1, xmax=1, color="blue")]
        try:
          cb = plt.colorbar(CF_prec, fraction=0.046, pad=0.01, aspect=25, label ="1h acc. prec.", extend="both")
        except:
          pass
        lg = ax1.legend(proxy, [f"Wind strength [m/s] at {dmap_meps.pressure[plev2]:.0f} hPa",
                               f"Geopotential [{tmap_meps.units.geopotential_pl}] at {dmap_meps.pressure[plev2]:.0f} hPa"])
        frame = lg.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(1)

      #if info:
        #  plt.text(x=0, y=-1, s="INFO: Reduced topographic noise by filtering with surface_geopotential bellow 3000",
        #           fontsize=7)  # , bbox=dict(facecolor='white', alpha=0.5))

      #plt.show()
      fig1.savefig("../../../output/{0}_Z500_VEL_P_{1}+{2:02d}.png".format(model, dt, ttt), bbox_inches="tight", dpi=200)
      ax1.cla()
      plt.clf()


    proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="green"),
             plt.axhline(y=0, xmin=1, xmax=1, color="blue")]
    fig2 = plt.figure(figsize=(2, 1.25))
    fig2.legend(proxy, [f"Wind strength [m/s] at {tmap_meps.pressure[plev2]:.0f} hPa",
                              f"Geopotential [{tmap_meps.units.geopotential_pl}]{tmap_meps.pressure[plev2]:.0f} hPa"])

    fig2.savefig("../../../output/{0}_Z500_VEL_P_LEGEND.png".format(model), bbox_inches="tight", dpi=200)

    try:
      fig3, ax3 = plt.subplots()
      fig3.colorbar(CF_prec, fraction=0.046, pad=0.04)
      ax3.remove()
      fig3.savefig("../../../output/{0}_Z500_VEL_P_COLORBAR.png".format(model), bbox_inches="tight", dpi=200)
    except:
      pass
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
