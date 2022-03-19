# %%
# python dxs.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway
# 2020-11-24: Modified Z500_VEL.py to plot dxs

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
import matplotlib as mpl
from matplotlib.axes import Axes
from cartopy.vector_transform import vector_scalar_to_grid
#import metpy.calc as mpcalc

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
from add_overlays import *

def surf(datetime, steps=0, model=model, domain_name=None, domain_lonlat=None, legend=False, info=False,grid=True, runid=None, outpath=None):
    global OUTPUTPATH
    if outpath != None:
        OUTPUTPATH=outpath
    
    for dt in datetime: #modelrun at time..
        if runid !=None:
            make_modelrun_folder = setup_directory( OUTPUTPATH, "{0}-{1}".format(dt,runid) )
        else:
            make_modelrun_folder = setup_directory( OUTPUTPATH, "{0}".format(dt) )

        date = dt[0:-2]
        hour = int(dt[-2:])
        # param_sfc = ["relative_humidity_2m", "air_temperature_2m", "specific_humidity_2m", "air_pressure_at_sea_level"]
        param_sfc = ["air_temperature_2m", "specific_humidity_2m", "air_pressure_at_sea_level", "surface_geopotential"]
        param_sfx = ["SFX_SST", "SFX_SIC"]
        param_pl = []
        param = param_sfc + param_pl
        split = False
        print("\n######## Checking if your request is possibel ############")
        try:
            check_all = check_data(date=dt, model=model, param=param, step=steps)
            check_sfx = check_data(date=dt, model=model, param=param_sfx, step=steps)

            print(check_all.file)

        except ValueError:
            print("!!!!! Sorry this plot is not availbale for this date. Try with another datetime !!!!!")
            break
        print("--------> Found match for your request ############")

        if not split:
            file_all = check_all.file.loc[0]
            file_sfx = check_sfx.file.loc[0]

            data_domain = domain_input_handler(dt, model, domain_name, domain_lonlat, file_all)

            # lonlat = np.array(data_domain.lonlat)
            print(file_all)
            dmap_meps = get_data(model=model, data_domain=data_domain, param=param, file=file_all, step=steps,
                                 date=dt)
            dmap_meps_sfx = get_data(model=model, data_domain=data_domain, param=param_sfx, file=file_sfx, step=steps,
                                     date=dt)
            print("\n######## Retriving data ############")
            print(f"--------> from: {dmap_meps.url} ")
            dmap_meps.retrieve()
            tmap_meps = dmap_meps  # two names for same value, no copying done.
            dmap_meps_sfx.retrieve()

        # convert fields
        # dmap_meps.air_pressure_at_sea_level/=100
        # u,v = xwind2uwind(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m, tmap_meps.alpha)
        # vel = wind_speed(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m)

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
        # crs = ccrs.PlateCarree()

        for tim in np.arange(np.min(steps), np.max(steps) + 1, 1):

            # determine if image should be created for this time step
            stepok=False
            if tim<25:
                stepok=True
            elif (tim<=48) and ((tim % 3) == 0):
                stepok=True
            elif (tim<=66) and ((tim % 6) == 0):
                stepok=True
            if stepok==True:

                fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9),subplot_kw={'projection': crs})
                ttt = tim  # + np.min(steps)
                tidx = tim - np.min(steps)
                print('Plotting d-excess {0} + {1:02d} UTC'.format(dt, ttt))
                SI = dmap_meps_sfx.SFX_SIC[tidx, :, :].squeeze()
                SImask = np.where(SI >= 0.1, dmap_meps_sfx.SFX_SIC[tidx, :, :], np.NaN).squeeze()

                ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
                MSLP = np.where(ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze() / 100
                # print("SIC: {}, {}".format(min(MSLP[0][:]), max(MSLP[0][:])))
                # SST = np.where(SI == 0, dmap_meps_sfx.SST[tidx, :, :], np.NaN).squeeze()
                # TP = precip_acc(dmap_meps.precipitation_amount_acc, acc=1)[tidx, 0, :,:].squeeze()
                # L = dmap_meps_sfx.LE[tidx,:,:].squeeze()
                # L = np.where(ZS < 3000, L, np.NaN).squeeze()
                # SH = dmap_meps_sfx.H[tidx,:,:].squeeze()
                # SH = np.where(ZS < 3000, SH, np.NaN).squeeze()
                SST = dmap_meps_sfx.SFX_SST[tidx, :, :].squeeze() - 273.15
                es = 6.1094 * np.exp(17.625 * SST / (SST + 243.04))
                mslp = dmap_meps.air_pressure_at_sea_level[tidx, :, :].squeeze() / 100
                if model == 'AromeArctic':
                    Q = dmap_meps.specific_humidity_2m[tidx, :, :].squeeze()
                else:
                    #Q = 1
                    Q = dmap_meps.specific_humidity_2m[tidx, :, :].squeeze()
                    # AT = dmap_meps.air_temperature_2m[tidx,:,:].squeeze()
                    # w =
                    # Q = w / (w+1)
                Qs = 0.622 * es / (mslp - 0.37 * es)
                # RH_2m = dmap_meps.relative_humidity_2m[tidx,:,:].squeeze()*100
                RH = Q / Qs * 100
                # print("RH: {}".format(RH_2m[0][0]))
                d = 48.2 - 0.54 * RH
                # Ux = dmap_meps.x_wind_10m[tidx, 0, :, :].squeeze()
                # Vx = dmap_meps.y_wind_10m[tidx, 0, :, :].squeeze()
                # xm,ym = np.meshgrid(dmap_meps.x, dmap_meps.y)

                # VELOCITY
                # new_x, new_y, new_u, new_v, = vector_scalar_to_grid(src_crs= data, target_proj= crs_lon,regrid_shape = np.shape(Ux), x= dmap_meps.x, y= dmap_meps.y, u= Ux, v= Vx)
                # magnitude = (new_u ** 2 + new_v ** 2) ** 0.5
                # cmap = plt.get_cmap("viridis") #cividis copper
                # wii = Axes.streamplot(ax1, new_x, new_y, new_u, new_v, density=4,zorder=4,transform=crs_lon, linewidth=0.7, color=magnitude, cmap=cmap)
                # LATENT
                # levelspos=np.arange(80, round(np.nanmax(L), -1) + 10, 40)
                # levelsneg = np.arange(-300, -9, 10)
                # levels = np.append(levelsneg, levelspos)
                # CL = ax1.contour(dmap_meps.x, dmap_meps.y, L, zorder=3, alpha=1.0, colors="red", linewidths=0.7, levels=levels, transform=data)
                # ax1.clabel(CL, CL.levels[::2], inline=True, fmt="%3.0f", fontsize=10)
                # xx = np.where(L < -10, xm, np.NaN).squeeze()
                # yy = np.where(L < -10, ym, np.NaN).squeeze()
                # skip = (slice(None, None, 4), slice(None, None, 4))
                # ax1.scatter(xx[skip][skip], yy[skip][skip], s=20, zorder=2, marker='x', linewidths=0.9,
                #                         c="white", alpha=0.7, transform=data)
                # xx = np.where(L > 80, xm, np.NaN).squeeze()
                # yy = np.where(L > 80, ym, np.NaN).squeeze()
                # skip = (slice(None, None, 4), slice(None, None, 4))
                # ax1.scatter(xx[skip][skip], yy[skip][skip], s=20, zorder=2, marker='.', linewidths=0.9,
                #             c="black", alpha=0.7, transform=data)

                # SENSIBLE
                # levelspos = np.arange(80, round(np.nanmax(SH), -1) + 10, 40)
                # levelsneg = np.arange(-300, -9, 10)
                # levels = np.append(levelsneg, levelspos)
                # CSH = ax1.contour(dmap_meps.x, dmap_meps.y, SH, zorder=3, alpha=1.0, colors="blue", linewidths=0.7, levels=levels, transform=data)
                # ax1.clabel(CSH, CSH.levels[1::2], inline=True, fmt="%3.0f", fontsize=10)
                # xx = np.where(SH < -10, xm, np.NaN).squeeze()
                # yy = np.where(SH < -10, ym, np.NaN).squeeze()
                # skip = (slice(None, None, 4), slice(None, None, 4))
                # ax1.scatter(xx[skip][skip],yy[skip][skip],s=20, zorder=2, marker='x',linewidths=0.9, c= "white", alpha=0.7, transform=data)

                # xx = np.where(SH >80, xm, np.NaN).squeeze()
                # yy = np.where(SH >80, ym, np.NaN).squeeze()
                # skip = (slice(None, None, 4), slice(None, None, 4))
                # ax1.scatter(xx[skip][skip], yy[skip][skip], s=20, zorder=2, marker='.', linewidths=0.9, c="black", alpha=0.7,
                #             transform=data)

                # SST
                levels = np.arange(-10, 45, 5)
                cmap = plt.get_cmap("cividis_r")
                Cd = ax1.contourf(dmap_meps.x, dmap_meps.y, d, zorder=1, alpha=0.7, cmap=cmap, levels=levels, extend="both",
                                  transform=data)
                Cd10 = ax1.contour(dmap_meps.x, dmap_meps.y, d, zorder=1, alpha=0.7, colors='tab:blue', levels=[10],
                                   transform=data)
                SI = ax1.contourf(dmap_meps.x, dmap_meps.y, SImask, zorder=2, alpha=1, colors='azure', transform=data)
                ax1.contour(dmap_meps.x, dmap_meps.y, SImask, zorder=2, alpha=1, colors='black', levels=[0.15],
                            transform=data, linestyles='--')

                C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=4, alpha=1.0,
                                  levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 2),
                                  colors='dimgrey', linewidths=0.5)
                C_P = ax1.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=4, alpha=1.0,
                                  levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 10),
                                  colors='dimgrey', linewidths=1.0)
                ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

                ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'), zorder=3,
                                facecolor="whitesmoke")  
                ax1.text(0, 1, "{0}_dxs_{1}+{2:02d}".format(model, dt, ttt), ha='left', va='bottom', \
                         transform=ax1.transAxes, color='black')
                ##########################################################
                # handles, labels = ax1.get_legend_handles_labels()
                legend = True
                if legend:
                    proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="tab:blue"), \
                             plt.axhline(y=0, xmin=1, xmax=1, color="dimgrey", linewidth=0.5), \
                             plt.axhline(y=0, xmin=1, xmax=1, color="black", linestyle='--'), \
                             ]
                    lg = plt.legend(proxy, ["d-xs=10", "MSLP [hPa]", "Sea Ice conc. >10%"], loc=1)


                    # cb = plt.colorbar(CSST, fraction=0.046, pad=0.01, ax=ax1, aspect=25, label ="RH [%]", extend = "both")
                    ax_cb = adjustable_colorbar_cax(fig1, ax1)
                    cb = plt.colorbar(Cd, fraction=0.046, pad=0.01, ax=ax1, aspect=25, cax= ax_cb, label="d-excess ($\perthousand$)",
                                      extend="both")

                    frame = lg.get_frame()
                    frame.set_facecolor('lightgray')
                    frame.set_alpha(1)
                print("filename: " + make_modelrun_folder + "/{0}_{1}_dxs_{2}+{3:02d}.png".format(model, domain_name, dt,
                                                                                                 ttt))
                if grid:
                    nicegrid(ax=ax1)

                add_ISLAS_overlays(ax1)

                if domain_name != model and data_domain != None:  # weird bug.. cuts off when sees no data value
                    ax1.set_extent(data_domain.lonlat)
                fig1.savefig(make_modelrun_folder + "/{0}_{1}_dxs_{2}+{3:02d}.png".format(model, domain_name, dt, ttt),
                             bbox_inches="tight", dpi=200)
                ax1.cla()
                plt.clf()
                plt.close(fig1)
    plt.close("all")

#

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
  parser.add_argument("--steps", default=0, nargs="+", type=int,
                        help="forecast times example --steps 0 3 gives time 0 to 3")
  parser.add_argument("--model", default="MEPS", help="MEPS or AromeArctic")
  parser.add_argument("--domain_name", default=None, help="MEPS or AromeArctic")
  parser.add_argument("--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--legend", default=False, help="Display legend")
  parser.add_argument("--grid", default=True, help="Display legend")
  parser.add_argument("--info", default=False, help="Display info")
  parser.add_argument("--id", default=None, help="Display legend", type=str)
  parser.add_argument("--outpath", default=None, help="Display legend", type=str)
  args = parser.parse_args()

  surf(datetime=args.datetime, steps = args.steps, model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info, grid=args.grid,  runid =args.id, outpath=args.outpath)

# fin
