from weathervis.config import *
from weathervis.utils import *
from weathervis.checkget_data_handler import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
#import weathervis.config as wc
import cartopy.crs as ccrs
from weathervis.domain import *  # require netcdf4
from weathervis.check_data import *
from weathervis.get_data import *
from weathervis.calculation import *
import matplotlib.pyplot as plt
import warnings
import cartopy.feature as cfeature
import netCDF4 as nc
import matplotlib.colors as colors
import datetime as dt_m

#####################################################################################################################
def flexpart_EC(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None,
                legend=False, info = False, save = True, grid=True, flex_base_path="", track = False):
  for dt in datetime: #modelrun at time..
    dt = f"{dt}"
    date = dt[0:-2]
    hour = int(dt[-2:])
    print(date)
    param = ["air_pressure_at_sea_level", "surface_geopotential"]
    print(steps)
    print(domain_name)
    dmap_meps, dom_name, bad_param = checkget_data_handler(domain_name=domain_name, all_param= param, date=dt, model = model, step=steps)
    # convert fields
    dmap_meps.air_pressure_at_sea_level /= 100
    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    parallels = dmap_meps.standard_parallel_projection_lambert
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
    crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,globe=globe)
    #release_name = 'NYAlesund_S1'
    #path = "{0}/{1}/flexpart_run_d01_combined.nc".format(flex_base_path, release_name)
    #path = "/Users/ainajoh/Downloads/here.nc"
    #path = "/Users/ainajoh/Downloads/cmet1.nc"
    print(path)
    #findpath = glob.glob(path)
    #print(findpath)
    cdf = nc.Dataset(path, "r")  # "/home/centos/flexpart/{0}/grid_conc_{1}0000.nc".format(release_name,dt), "r")
    lats = cdf.variables["XLAT"][:]
    lons = cdf.variables["XLONG"][:]
    #lons, lats = np.meshgrid(lons, lats)
    tim_data = cdf.variables["time"][:]
    levs = cdf.variables["ZTOP"][:]
    spec1a = cdf.variables["CONC"][:]
    print(levs)
    print(np.shape(spec1a))
    print(len(spec1a))
    print(len(np.shape(spec1a)))
    make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
    #time_read = dt_m.datetime.utcfromtimestamp(tim_data)

    print(len(tim_data))
    for tim in np.arange(np.min(steps), np.max(steps)+1,1):
        ax1 = plt.subplot(projection=crs)
        epoch_now = tim_data[tim]
        time_read = dt_m.datetime.utcfromtimestamp(epoch_now)

        stepok = False
        if tim < 0:  # do not need hourly steps for FP
            stepok = True
        elif (tim <= 36) and ((tim % 3) == 0):
            stepok = True
        elif (tim <= 120) and ((tim % 3) == 0):
            stepok = True

        if stepok==True:#
            l=0
            for lev in levs[:]:
                fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9),subplot_kw={'projection': crs})
                ttt = tim
                tidx = tim - np.min(steps)
                if len(np.shape(spec1a)) == 5:
                    print("hereee if many releases")
                    spec2a=(spec1a[tim, :, l, :, :]).squeeze()
                else:
                    spec2a = np.sum(spec1a[tim, l, :, :], 0).squeeze()
                spec2a = np.where(spec2a > 1e-10, spec2a, np.NaN)
                print('Plotting FLEXPART-EC {0} + {1:02d} UTC, level {2}'.format(dt,tim,lev))
                plev = 0
                Z = dmap_meps.surface_geopotential[tidx, 0, :, :]
                MSLP = np.where(Z < 50000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()
                C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=6, alpha=1.0,
                                    levels=np.arange(960, 1050, 1), colors='grey', linewidths=0.5, transform=crs)
                C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=7, alpha=1.0,
                                    levels=np.arange(960, 1050, 10),
                                    colors='grey', linewidths=1.0, label="MSLP [hPa]", transform=crs)
                ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

                c = ['#F7A7FD','#A7D3FD','#FDDBA7',
                     '#00FF80', '#606060','#9933FF']

                #cmap= [plt.cm.Reds,  plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples]
                cmapmy= ["Reds", "Blues", "Oranges","Greens", "Greys", "Purples"]

                colorindx = 0
                label_leg = []
                for rel in range(0,np.shape(spec2a)[0]):
                    print(colorindx)
                    print(c[colorindx])
                    cm = get_continuous_cmap( [ c[colorindx], c[colorindx] ] )
                    cm = cmapmy[colorindx]

                    FP = ax1.pcolormesh(lons, lats, spec2a[rel,:,:],  norm=colors.LogNorm(vmin=1e-10, vmax=0.2), cmap=cm, zorder=1, alpha=0.6, transform=ccrs.PlateCarree())
                    #levels = np.linspace(1e-10,0.2,3)
                    #levels = [0, 1e-10, 0.1]
                    #if rel==0:
                    #    fPC = ax1.contour(lons, lats, spec2a[rel,:,:], cmap=cm, zorder=2, levels= levels, alpha=1, transform=ccrs.PlateCarree())
                    #if rel==1:
                    #    fPC = ax1.contour(lons, lats, spec2a[rel, :, :], cmap=cm, zorder=2, levels=levels, alpha=1,linestyle="--",
                    #                      transform=ccrs.PlateCarree())
                    labeltext = f"rel_{rel}"
                    #tot_patch = mpl.patches.Patch(color='gray', alpha=0.5, linewidth=0)
                    label_leg += [Line2D([0], [0], marker="s", color=cm[:-1], label=labeltext, markersize=15,lw=0)]
                    colorindx += 1

                ax1.legend(handles=label_leg, loc='upper left').set_zorder(99999)
                ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'))
                ax1.text(0, 1, "{0}_FP_{1}+{2:02d}".format(model, dt, ttt), ha='left', va='bottom', transform=ax1.transAxes, color='black')

                if track:
                    tt = dmap_meps.time[tim]
                    plot_track_on_map(dt=dt, model=model, tim=tim, gca=plt.gca(), ccrs=ccrs, c1="gray", c2="red", tt=tt, url="/Users/ainajoh/Downloads/Data_210327_0545Z")
                if grid:
                    nicegrid(ax=ax1)

                # if domain_name != model and data_domain != None:  # weird bug.. cuts off when sees no data value
                lonlat = [dmap_meps.longitude[0, 0], dmap_meps.longitude[-1, -1], dmap_meps.latitude[0, 0],
                          dmap_meps.latitude[-1, -1]]
                ax1.set_extent(lonlat)
                lev_in=int(lev)
                lev_in=f"{lev_in}"
                lev_in=lev_in.zfill(5)
                fig1.savefig(
                    make_modelrun_folder + "/FLEXPART_AA_{0}_L{1}_{2}+{3:02d}.png".format(domain_name, lev_in, dt, tim),
                    bbox_inches="tight", dpi=200)
                l+=1
                ax1.cla()
                plt.clf()
                plt.close(fig1)


def old():
    file= "/Users/ainajoh/multirel4cmet.nc"
    cdf = nc.Dataset(file, "r")  # "/home/centos/flexpart/{0}/grid_conc_{1}0000.nc".format(release_name,dt), "r")
    print(cdf)
    lats=cdf.variables["XLAT"][:]
    lons=cdf.variables["XLONG"][:]
    tim=cdf.variables["time"][:]
    levs=cdf.variables["ZTOP"][:]
    spec1a=cdf.variables["CONC"]
    #lons, lats = np.meshgrid(lons, lats)
    print(np.shape(spec1a))
    print(np.shape(lats))
    print(np.shape(dmap_meps.air_pressure_at_sea_level))
    print(levs)
    spec1a = spec1a[30,0,1,:,:]
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})
    F_P = ax1.pcolormesh(lons, lats, spec1a, cmap=plt.cm.Reds, zorder=1,
                     alpha=0.9, transform=ccrs.PlateCarree())
    C_P = ax1.contour(dmap_meps.x, dmap_meps.y, dmap_meps.air_pressure_at_sea_level[0, 0,:,:], colors='grey', linewidths=0.5)
    ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'), alpha = 0.2)

    plt.show()


if __name__ == "__main__":
  import argparse

  def none_or_str(value):
    if value == 'None':
      return None
    return value

  parser = argparse.ArgumentParser()
  parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+", type=str)
  parser.add_argument("--steps", default=0, nargs="+", type=int,help="forecast times example --steps 0 3 gives time 0 to 3")
  parser.add_argument("--model",default="AromeArctic", help="MEPS or AromeArctic")
  parser.add_argument("--domain_name", default=None, help="see domain.py", type = none_or_str)
  parser.add_argument("--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--legend", default=False, help="Display legend")
  parser.add_argument("--grid", default=True, help="Display legend")
  parser.add_argument("--info", default=False, help="Display info")
  parser.add_argument("--track", default=False, help="Display info", type=bool)

  args = parser.parse_args()
  print(args.__dict__)

  #flex_base_path= "/home/centos/flexpart-arome/{0}".format(dt)
  flex_base_path= "/Data/gfi/work/cat010/flexpart_arome/output/{0}".format(args.datetime[0])


  # split up in 3 retrievals of up to 24h
  flexpart_EC(datetime=args.datetime, steps = [np.min(args.steps), np.max(args.steps)], model = args.model, domain_name = args.domain_name,
         domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info, grid=args.grid, flex_base_path=flex_base_path, track= args.track)
  #old()
#fin
