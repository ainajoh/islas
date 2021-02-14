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

def surf(datetime, steps=0, model= "AromeArctic", domain_name = None, domain_lonlat = None, legend=False, info = False,grid=True):

  for dt in datetime: #modelrun at time..
    date = dt[0:-2]
    hour = int(dt[-2:])
    param_sfc = ["surface_geopotential","air_pressure_at_sea_level", "x_wind_10m", "y_wind_10m","precipitation_amount_acc", "wind_speed"]
    param_sfx = ["LE", "H", "SST"]
    param_pl = []
    param = param_sfc + param_pl
    split = False
    sfx=False
    print("\n######## Checking if your request is possible ############")
    try:
      check_all = check_data(date=dt, model=model, param=param, step=steps)
      check_sfx = check_data(date=dt, model=model, param=param_sfx,step=steps)

      print(check_all.file)

    except ValueError:
        try:
          sfx = True
          param_sfx = ["SFX_LE", "SFX_H", "SFX_SST"]
          check_all = check_data(date=dt, model=model, param=param, step=steps)
          check_sfx = check_data(date=dt, model=model, param=param_sfx, step=steps)
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
      print("\n######## Retrieving data ############")
      print(f"--------> from: {dmap_meps.url} ")
      dmap_meps.retrieve()
      tmap_meps = dmap_meps # two names for same value, no copying done.
      dmap_meps_sfx.retrieve()
      if sfx:#["SFX_LE", "SFX_H", "SFX_SST"]
        dmap_meps_sfx.LE = dmap_meps_sfx.SFX_LE
        dmap_meps_sfx.H = dmap_meps_sfx.SFX_H
        dmap_meps_sfx.SST = dmap_meps_sfx.SFX_SST
        dmap_meps_sfx.units.LE = dmap_meps_sfx.units.SFX_LE
        dmap_meps_sfx.units.H = dmap_meps_sfx.units.SFX_H
        dmap_meps_sfx.units.SST = dmap_meps_sfx.units.SFX_SST

    # convert fields
    dmap_meps.air_pressure_at_sea_level/=100
    u,v = xwind2uwind(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m, tmap_meps.alpha)
    vel = wind_speed(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m)

    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    parallels = dmap_meps.standard_parallel_projection_lambert

    # setting up projection
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
    data = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                 globe=globe)
    crs = data
    crs_lon = ccrs.PlateCarree()

    make_modelrun_folder = setup_directory( OUTPUTPATH, "{0}".format(dt) )
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9),
                               subplot_kw={'projection': crs})

    #crs = ccrs.PlateCarree()
    for tim in np.arange(np.min(steps), np.max(steps)+1, 1):
      ttt = tim #+ np.min(steps)
      tidx = tim - np.min(steps)
      print('Plotting {0} + {1:02d} UTC'.format(dt, ttt))
      ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
      MSLP = np.where(ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()
      #TP = precip_acc(dmap_meps.precipitation_amount_acc, acc=1)[tidx, 0, :,:].squeeze()
      L = dmap_meps_sfx.LE[tidx,:,:].squeeze()
      L = np.where(ZS < 3000, L, np.NaN).squeeze()
      SH = dmap_meps_sfx.H[tidx,:,:].squeeze()
      SH = np.where(ZS < 3000, SH, np.NaN).squeeze()
      SST = dmap_meps_sfx.SST[tidx,:,:].squeeze()
      Ux = dmap_meps.x_wind_10m[tidx, 0, :, :].squeeze()
      Vx = dmap_meps.y_wind_10m[tidx, 0, :, :].squeeze()
      xm,ym = np.meshgrid(dmap_meps.x, dmap_meps.y)
      uxx = dmap_meps.x_wind_10m[tidx, 0, :, :].squeeze()
      vxx = dmap_meps.y_wind_10m[tidx, 0, :, :].squeeze()

      #VELOCITY
      #new_x, new_y, new_u, new_v, = vector_scalar_to_grid(src_crs= data, target_proj= crs_lon,regrid_shape = np.shape(Ux), x= dmap_meps.x, y= dmap_meps.y, u= Ux, v= Vx)
      #magnitude = (new_u ** 2 + new_v ** 2) ** 0.5
      #cmap = plt.get_cmap("viridis") #cividis copper
      #wii = plt.quiver(new_x, new_y, new_u, new_v)
      #wii = ax1.quiver(xm, ym, Ux, Vx)

      #wii = Axes.streamplot(ax1, new_x, new_y, new_u, new_v, density=4,zorder=4,transform=crs_lon, linewidth=0.7, color=magnitude, cmap=cmap)

      #LATENT
      #levelspos=np.arange(80, round(np.nanmax(L), -1) + 10, 40)
      #levelsneg = np.arange(-300, -9, 10)
      #levels = np.append(levelsneg, levelspos)
      #CL = ax1.contour(dmap_meps.x, dmap_meps.y, L, zorder=3, alpha=1.0, colors="red", linewidths=0.7, levels=levels, transform=data)
      #ax1.clabel(CL, CL.levels[::2], inline=True, fmt="%3.0f", fontsize=10)
      #xx = np.where(L < -10, xm, np.NaN).squeeze()
      #yy = np.where(L < -10, ym, np.NaN).squeeze()
      #skip = (slice(None, None, 4), slice(None, None, 4))
      #ax1.scatter(xx[skip][skip], yy[skip][skip], s=20, zorder=2, marker='x', linewidths=0.9,
      #                        c="white", alpha=0.7, transform=data)
      #xx = np.where(L > 80, xm, np.NaN).squeeze()
      #yy = np.where(L > 80, ym, np.NaN).squeeze()
      #skip = (slice(None, None, 4), slice(None, None, 4))
      #ax1.scatter(xx[skip][skip], yy[skip][skip], s=20, zorder=2, marker='.', linewidths=0.9,
      #            c="black", alpha=0.7, transform=data)

      #xx = np.where(SH >80, xm, np.NaN).squeeze()
      #yy = np.where(SH >80, ym, np.NaN).squeeze()
      #skip = (slice(None, None, 4), slice(None, None, 4))
      #ax1.scatter(xx[skip][skip], yy[skip][skip], s=20, zorder=2, marker='.', linewidths=0.9, c="black", alpha=0.7,
      #            transform=data)

      #SST_new
      #levels=np.arange(270,294,2)
      SST = SST - 273.15
      #levels = [np.min(SST), np.max(SST), 3]
      levels = [0,2,4,6,8,10,12,15,18,21,24]
      C_SS = ax1.contour(dmap_meps.x, dmap_meps.y, SST, colors="darkred", linewidths=2, levels =levels, zorder=8)
      ax1.clabel(C_SS, C_SS.levels, inline=True, fmt="%3.0f", fontsize=10 )

      #MSLP
      #MSLP with contour labels every 10 hPa
      C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=10, alpha=1.0,
                        levels=np.arange(960, 1050, 1),
                        colors='grey', linewidths=0.5)
      C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=10, alpha=1.0,
                        levels=np.arange(960, 1050, 10),
                        colors='grey', linewidths=1.0, label="MSLP [hPa]")
      ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

      #wind#
      #skip = (slice(50, -50, 50), slice(50, -50, 50))
      skip = (slice(10, -10, 30), slice(10, -10, 30)) #70
      scale = 1.94384
      CVV = ax1.barbs(xm[skip], ym[skip], uxx[skip]*scale, vxx[skip]*scale, length=5.5, zorder=11)

      #lat_p = 60.2
      #lon_p = 5.4167
      #mainpoint = ax1.scatter(lon_p, lat_p, s=9.0 ** 2, transform=ccrs.PlateCarree(),
      #                        color='lime', zorder=6, linestyle='None', edgecolors="k", linewidths=3)

      #LATENT_new
      #levels=np.arange(270,294,2)
      cmap = plt.get_cmap("coolwarm")
      levels = np.linspace(-150,200,8)

      CLH = ax1.contourf(dmap_meps.x, dmap_meps.y, L, zorder=1, levels=levels, alpha=0.7, cmap = cmap, extend = "both", transform=data)
      ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),zorder=3,facecolor="white",edgecolor="gray")  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
      ax1.text(0, 1, "{0}_surf_{1}+{2:02d}".format(model, dt, ttt), ha='left', va='bottom', transform=ax1.transAxes, color='black')
      ##########################################################
      #handles, labels = ax1.get_legend_handles_labels()

      #SENSIBLE
      #levelspos = np.arange(20, round(np.nanmax(SH), -10) + 10, 40)
      #levelsneg = np.arange(-300, -19, 40)
      #levels = np.append(levelsneg, levelspos)
      #levels = np.linspace(-300,300,15)
      #CSH = plt.contour(dmap_meps.x, dmap_meps.y, SH, alpha=1.0, colors="blue", linewidths=0.7, levels=levels, zorder=14)
      CSH = plt.contour(dmap_meps.x, dmap_meps.y, SH, alpha=1.0, colors="blue", linewidths=0.7, zorder=14)
      #ax1.clabel(CSH, CSH.levels[1::2], inline=True, fmt="%3.0f", fontsize=10)
      #xx = np.where(SH < -10, xm, np.NaN).squeeze()
      #yy = np.where(SH < -10, ym, np.NaN).squeeze()
      #skip = (slice(None, None, 4), slice(None, None, 4))
      #ax1.scatter(xx[skip][skip],yy[skip][skip],s=20, zorder=2, marker='x',linewidths=0.9, c= "white", alpha=0.7, transform=data)

      legend=True
      if legend:
        proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="blue"),
                 plt.axhline(y=0, xmin=1, xmax=1, color="darkred")]
        lg = plt.legend(proxy, [f"Sensible heat [{dmap_meps_sfx.units.H}] ", f"SST [C]"],loc=1)
        ax_cb = adjustable_colorbar_cax(fig1, ax1)
        cb = plt.colorbar(CLH, cax= ax_cb, fraction=0.046, pad=0.01, ax=ax1, aspect=25, label =f"Latent heat [{dmap_meps_sfx.units.LE}]", extend = "both")
        frame = lg.get_frame()
        lg.set_zorder(102)
        frame.set_facecolor('lightgray')
        frame.set_alpha(1)
        #plt.title("{0}_surf_{1}+{2:02d}.png".format(model, dt, ttt))
      #print("test")
      #print(OUTPUTPATH)
      #print("{0}".format(dt))
      if grid:
        nicegrid(ax=ax1)
      if domain_name != model and data_domain != None:  # weird bug.. cuts off when sees no data value
        ax1.set_extent(data_domain.lonlat)
      fig1.savefig(make_modelrun_folder + "/{0}_{1}_surf_{2}+{3:02d}.png".format(model, domain_name, dt, ttt), bbox_inches="tight", dpi=200)

      ax1.cla()
      #cb.remove()
      #lg.remove()
      #plt.draw()

    plt.clf()
    plt.close(fig1)
  plt.close("all")
# fin

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
  surf(datetime=args.datetime, steps = args.steps, model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info,grid=args.grid)
  #datetime, step=4, model= "MEPS", domain = None
# ax1.fill(xx[skip][skip], yy[skip][skip], color="none", hatch='X', edgecolor="b", linewidth=0.0)
