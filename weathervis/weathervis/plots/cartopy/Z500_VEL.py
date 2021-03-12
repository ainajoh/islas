# %%
# python Z500_VEL.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway

from weathervis.config import *
from weathervis.utils import *
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

def Z500_VEL(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None, legend=False, info = False,grid=True):

  for dt in datetime: #modelrun at time..
    date = dt[0:-2]
    hour = int(dt[-2:])
    param_sfc = ["air_pressure_at_sea_level", "precipitation_amount_acc", "surface_geopotential"]
    param_pl = ["x_wind_pl", "y_wind_pl", "geopotential_pl"]
    param = param_sfc + param_pl
    plevel = [500]
    split = False
    print("\n######## Checking if your request is possible ############")
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
      file_pl = check_pl.file.loc[0]
      tmap_meps = get_data(model=model, data_domain=data_domain, param=param_pl, file=file_pl, step=steps, date=dt, p_level=plevel)
      print("\n######## Retrieving data ############")
      print(f"--------> from: {tmap_meps.url} ")
      tmap_meps.retrieve()


    # convert fields
    dmap_meps.air_pressure_at_sea_level/=100
    #dmap_meps.precipitation_amount_acc*=1000.0
    print(dmap_meps.units.precipitation_amount_acc)
    tmap_meps.geopotential_pl/=10.0
    tmap_meps.units.geopotential_pl ="m"
    u,v = xwind2uwind(tmap_meps.x_wind_pl,tmap_meps.y_wind_pl, tmap_meps.alpha)
    vel = wind_speed(tmap_meps.x_wind_pl,tmap_meps.y_wind_pl)

    # plot map
    lonlat = [dmap_meps.longitude[0, 0], dmap_meps.longitude[-1, -1], dmap_meps.latitude[0, 0],
              dmap_meps.latitude[-1, -1]]

    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    parallels = dmap_meps.standard_parallel_projection_lambert

    # setting up projection
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
    crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                globe=globe)

    make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))

    for tim in np.arange(np.min(steps), np.max(steps)+1, 1):

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
          ttt = tim #+ np.min(steps)
          tidx = tim - np.min(steps)
          print('Plotting Z500 {0} + {1:02d} UTC'.format(dt, ttt))
          plev2 = 0
          embr = 0
          ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
          MSLP = np.where(ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()
          acc=1
          TP = precip_acc(dmap_meps.precipitation_amount_acc, acc=acc)[tidx, 0, :,:].squeeze()
          VEL = (vel[tidx, plev2, :, :]).squeeze()
          Z = (tmap_meps.geopotential_pl[tidx, plev2, :, :]).squeeze()
          Ux = u[tidx, 0,:, :].squeeze()
          Vx = v[tidx, 0,:, :].squeeze()
          uxx = tmap_meps.x_wind_pl[tidx, 0,:, :].squeeze()
          vxx = tmap_meps.y_wind_pl[tidx, 0,:, :].squeeze()
          cmap = plt.get_cmap("tab20c")
          lvl = [0.02, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 25, 30]
          norm = mcolors.BoundaryNorm(lvl, cmap.N)

          try: #workaround for a stupid matplotlib error not handling when all values are outside of range in lvl or all just nans..
            #https://github.com/SciTools/cartopy/issues/1290
            #cmap =  mcolors.ListedColormap('hsv', 'hsv') #plt.get_cmap("hsv")PuBu
            #TP.filled(np.nan) #fill mask with nan to avoid:  UserWarning: Warning: converting a masked element to nan.
            CF_prec = plt.contourf(dmap_meps.x, dmap_meps.y, TP, zorder=1,
                                   cmap=cmap, norm = norm, alpha=0.4, antialiased=True,
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
          ####REMOVE LATER 60.2;5.4166666666667;60;None;N


          skip=20
          skip = (slice(20, -20, 50), slice(20, -20, 50)) #70
          xm,ym=np.meshgrid(dmap_meps.x, dmap_meps.y)
          CVV = ax1.barbs( xm[skip], ym[skip], uxx[skip]*1.94384, vxx[skip]*1.94384, length=6.5, zorder=5)
          #CS = ax1.contour(dmap_meps.x, dmap_meps.y, VEL, zorder=3, alpha=1.0,
          #                   levels=np.arange(-80, 80, 5), colors="green", linewidths=0.7)
          # geopotential
          CS = ax1.contour(dmap_meps.x, dmap_meps.y, Z, zorder=3, alpha=1.0,
                            levels=np.arange(4600, 5800, 20), colors="blue", linewidths=0.7)
          ax1.clabel(CS, CS.levels, inline=True, fmt="%4.0f", fontsize=10)

          ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'))  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
          ax1.text(0, 1, "{0}_Z500_{1}+{2:02d}".format(model, dt, ttt), ha='left', va='bottom', transform=ax1.transAxes, color='black')
          if grid:
            nicegrid(ax=ax1)
          ##########################################################
          legend = True
          if legend:
            proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="gray"),
            plt.axhline(y=0, xmin=1, xmax=1, color="blue")]
            try:
              ax_cb = adjustable_colorbar_cax(fig1, ax1)

              cb = plt.colorbar(CF_prec, cax=ax_cb,fraction=0.046, pad=0.01, aspect=25, label =f"{acc}h acc. prec. [mm/{acc}h]", extend="both")

            except:
              pass
            lg = ax1.legend(proxy, [f"MSLP [hPa]",
                                   f"Geopotential height[{tmap_meps.units.geopotential_pl}] at {dmap_meps.pressure[plev2]:.0f} hPa"],
                            loc="upper right")
            frame = lg.get_frame()
            frame.set_facecolor('white')
            frame.set_alpha(1)

          #if info:
            #  plt.text(x=0, y=-1, s="INFO: Reduced topographic noise by filtering with surface_geopotential bellow 3000",
            #           fontsize=7)  # , bbox=dict(facecolor='white', alpha=0.5))

          #plt.show()
          if domain_name != model and data_domain != None:  # weird bug.. cuts off when sees no data value
            ax1.set_extent(data_domain.lonlat)
          fig1.savefig(make_modelrun_folder + "/{0}_{1}_Z500_VEL_P_{2}+{3:02d}.png".format(model, domain_name, dt, ttt), bbox_inches="tight", dpi=200)
          ax1.cla()
          plt.clf()
          plt.close(fig1)

        #proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="green"),
        #         plt.axhline(y=0, xmin=1, xmax=1, color="blue")]
        #fig2 = plt.figure(figsize=(2, 1.25))
        #fig2.legend(proxy, [f"Wind strength [m/s] at {tmap_meps.pressure[plev2]:.0f} hPa",
        #                          f"Geopotential [{tmap_meps.units.geopotential_pl}]{tmap_meps.pressure[plev2]:.0f} hPa"])
        #fig2.savefig(make_modelrun_folder+"/{0}_Z500_VEL_P_LEGEND.png".format(model), bbox_inches="tight", dpi=200)
        #plt.close(fig2)
        #try:
        #  fig3, ax3 = plt.subplots()
        #  fig3.colorbar(CF_prec, fraction=0.046, pad=0.04)
        #  ax3.remove()
        #  fig3.savefig(make_modelrun_folder+"/{0}_{1}_Z500_VEL_P_COLORBAR.png".format(model, domain_name), bbox_inches="tight", dpi=200)
        #  plt.close(fig3)
        #except:
        #  pass
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
  parser.add_argument("--info", default=False, help="Display info")
  args = parser.parse_args()

  Z500_VEL(datetime=args.datetime, steps = [0, np.min([24, np.max(args.steps)])], model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info, grid=args.grid)
  if np.max(args.steps)>24:
    Z500_VEL(datetime=args.datetime, steps = [27, np.min([36, np.max(args.steps)])], model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info, grid=args.grid)
  if np.max(args.steps)>36:
    Z500_VEL(datetime=args.datetime, steps = [42, np.max(args.steps)], model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info, grid=args.grid)

# fin
