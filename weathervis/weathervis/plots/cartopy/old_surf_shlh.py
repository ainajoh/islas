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
import matplotlib as mpl
from matplotlib.axes import Axes
from cartopy.vector_transform import vector_scalar_to_grid


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



def BL_state(datetime, steps=0, model= "AromeArctic", domain_name = None, domain_lonlat = None, legend=False, info = False):

  for dt in datetime: #modelrun at time..
    date = dt[0:-2]
    hour = int(dt[-2:])
    #param_sfc = ["surface_geopotential","air_pressure_at_sea_level", "x_wind_10m", "y_wind_10m", "air_temperature_0m", "air_temperature_2m","specific_humidity_2m", "relative_humidity_2m","precipitation_amount_acc","fog_area_fraction", "wind_speed", "wind_direction"]
    param_sfc = ["atmosphere_boundary_layer_thickness"]
    param_sfx = ["LE", "H", "SST"]
    param_pl = []
    param = param_sfc + param_pl
    split = False
    print("\n######## Checking if your request is possibel ############")
    try:
      check_all = check_data(date=dt, model=model, param=param, step=steps)
      check_sfx = check_data(date=dt, model=model, param=param_sfx,step=steps)

      print(check_all.file)

    except ValueError:
        print("!!!!! Sorry this plot is not availbale for this date. Try with another datetime !!!!!")
        break
    print("--------> Found match for your request ############")


    if not split:
      file_all = check_all.file.loc[0]
      file_sfx = check_sfx.file.loc[0]

      data_domain = domain_input_handler(dt, model,domain_name, domain_lonlat, file_all)

      #lonlat = np.array(data_domain.lonlat)
      print(file_all)
      dmap_meps = get_data(model=model, data_domain=data_domain, param=param, file=file_all, step=steps,
                           date=dt)
      dmap_meps_sfx = get_data(model=model, data_domain=data_domain, param=param_sfx, file=file_sfx, step=steps,
                           date=dt)
      print("\n######## Retriving data ############")
      print(f"--------> from: {dmap_meps.url} ")
      dmap_meps.retrieve()
      tmap_meps = dmap_meps # two names for same value, no copying done.
      dmap_meps_sfx.retrieve()


    # convert fields
    dmap_meps.air_pressure_at_sea_level/=100
    u,v = xwind2uwind(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m, tmap_meps.alpha)
    vel = wind_speed(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m)

    # plot map
    fig1 = plt.figure(figsize=(7, 9))

    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    parallels = dmap_meps.standard_parallel_projection_lambert

    # setting up projection
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
    data = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                 globe=globe)
    crs = data
    crs_lon = ccrs.PlateCarree()
    #crs = ccrs.PlateCarree()
    for tim in np.arange(np.min(steps), np.max(steps)+1, 1):
      ax1 = plt.subplot(projection=crs)

      ttt = tim #+ np.min(steps)
      tidx = tim - np.min(steps)
      print('Plotting {0} + {1:02d} UTC'.format(dt, ttt))
      ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
      #MSLP = np.where(ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()
      #TP = precip_acc(dmap_meps.precipitation_amount_acc, acc=1)[tidx, 0, :,:].squeeze()
      L = dmap_meps_sfx.LE[tidx,:,:].squeeze()
      L = np.where(ZS < 3000, L, np.NaN).squeeze()
      SH = dmap_meps_sfx.H[tidx,:,:].squeeze()
      SH = np.where(ZS < 3000, SH, np.NaN).squeeze()
      SST = dmap_meps_sfx.SST[tidx,:,:].squeeze()
      Ux = dmap_meps.x_wind_10m[tidx, 0, :, :].squeeze()
      Vx = dmap_meps.y_wind_10m[tidx, 0, :, :].squeeze()
      xm,ym = np.meshgrid(dmap_meps.x, dmap_meps.y)

      #VELOCITY
      new_x, new_y, new_u, new_v, = vector_scalar_to_grid(src_crs= data, target_proj= crs_lon,regrid_shape = np.shape(Ux), x= dmap_meps.x, y= dmap_meps.y, u= Ux, v= Vx)
      magnitude = (new_u ** 2 + new_v ** 2) ** 0.5
      cmap = plt.get_cmap("viridis") #cividis copper
      wii = Axes.streamplot(ax1, new_x, new_y, new_u, new_v, density=4,zorder=4,transform=crs_lon, linewidth=0.7, color=magnitude, cmap=cmap)
      #LATENT
      levelspos=np.arange(80, round(np.nanmax(L), -1) + 10, 40)
      levelsneg = np.arange(-300, -9, 10)
      levels = np.append(levelsneg, levelspos)
      CL = ax1.contour(dmap_meps.x, dmap_meps.y, L, zorder=3, alpha=1.0, colors="red", linewidths=0.7, levels=levels, transform=data)
      ax1.clabel(CL, CL.levels[::2], inline=True, fmt="%3.0f", fontsize=10)
      xx = np.where(L < -10, xm, np.NaN).squeeze()
      yy = np.where(L < -10, ym, np.NaN).squeeze()
      skip = (slice(None, None, 4), slice(None, None, 4))
      ax1.scatter(xx[skip][skip], yy[skip][skip], s=20, zorder=2, marker='x', linewidths=0.9,
                              c="white", alpha=0.7, transform=data)
      xx = np.where(L > 80, xm, np.NaN).squeeze()
      yy = np.where(L > 80, ym, np.NaN).squeeze()
      skip = (slice(None, None, 4), slice(None, None, 4))
      ax1.scatter(xx[skip][skip], yy[skip][skip], s=20, zorder=2, marker='.', linewidths=0.9,
                  c="black", alpha=0.7, transform=data)


      #SENSIBLE
      levelspos = np.arange(80, round(np.nanmax(SH), -1) + 10, 40)
      levelsneg = np.arange(-300, -9, 10)
      levels = np.append(levelsneg, levelspos)
      CSH = ax1.contour(dmap_meps.x, dmap_meps.y, SH, zorder=3, alpha=1.0, colors="blue", linewidths=0.7, levels=levels, transform=data)
      ax1.clabel(CSH, CSH.levels[1::2], inline=True, fmt="%3.0f", fontsize=10)
      xx = np.where(SH < -10, xm, np.NaN).squeeze()
      yy = np.where(SH < -10, ym, np.NaN).squeeze()
      skip = (slice(None, None, 4), slice(None, None, 4))
      ax1.scatter(xx[skip][skip],yy[skip][skip],s=20, zorder=2, marker='x',linewidths=0.9, c= "white", alpha=0.7, transform=data)

      xx = np.where(SH >80, xm, np.NaN).squeeze()
      yy = np.where(SH >80, ym, np.NaN).squeeze()
      skip = (slice(None, None, 4), slice(None, None, 4))
      ax1.scatter(xx[skip][skip], yy[skip][skip], s=20, zorder=2, marker='.', linewidths=0.9, c="black", alpha=0.7,
                  transform=data)

      #SST
      levels=np.arange(270,294,2)
      cmap = plt.get_cmap("coolwarm")
      CSST = ax1.contourf(dmap_meps.x, dmap_meps.y, SST, zorder=1, alpha=0.7, cmap = cmap, levels=levels, extend = "both", transform=data)


      ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),zorder=3,facecolor="whitesmoke")  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
      ##########################################################
      #handles, labels = ax1.get_legend_handles_labels()
      legend=True
      if legend:
        proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="blue"),
                 plt.axhline(y=0, xmin=1, xmax=1, color="red")]
        lg = plt.legend(proxy, [f"Sensible heat [{dmap_meps_sfx.units.H}] ", f"Latent heat [{dmap_meps_sfx.units.LE}]"],loc=1)

        cb = plt.colorbar(CSST, fraction=0.046, pad=0.01, ax=ax1, aspect=25, label ="SST [K]", extend = "both")
        frame = lg.get_frame()
        frame.set_facecolor('lightgray')
        frame.set_alpha(1)

      fig1.savefig("../../../output/{0}_surf_{1}+{2:02d}.png".format(model, dt, ttt), bbox_inches="tight", dpi=200)
      ax1.cla()
      #cb.remove()
      #lg.remove()
      #plt.draw()

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
  surf(datetime=args.datetime, steps = args.steps, model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info)
  #datetime, step=4, model= "MEPS", domain = None
# ax1.fill(xx[skip][skip], yy[skip][skip], color="none", hatch='X', edgecolor="b", linewidth=0.0)
