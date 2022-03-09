from weathervis.config import *
from weathervis.utils import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.check_data import *

from weathervis.calculation import *
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import matplotlib.patheffects as pe
from cartopy.io import shapereader  # For reading shapefiles containg high-resolution coastline.
from copy import deepcopy
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl
from weathervis.checkget_data_handler import *
from pylab import *
from add_overlays import *

def IVT_IWV(datetime, steps, model, domain_name = None, domain_lonlat = None, legend=False, info = False,grid=True,  m_level = [0, 64], runid=None, outpath=None):
    global OUTPUTPATH
    if outpath != None:
        OUTPUTPATH=outpath
        
    for dt in datetime: #modelrun at time..
        if runid !=None:
            make_modelrun_folder = setup_directory( OUTPUTPATH, "{0}-{1}".format(dt,runid) )
        else:
            make_modelrun_folder = setup_directory( OUTPUTPATH, "{0}".format(dt) )
    for dt in datetime:
        date = dt[0:-2]
        hour = int(dt[-2:])
        all_param = ['specific_humidity_ml','air_pressure_at_sea_level','surface_geopotential','surface_air_pressure','x_wind_ml','y_wind_ml']

        # loop over pressure levels, do not forget the indent the whole routine

        dmet,data_domain,bad_param = checkget_data_handler(all_param=all_param, date=dt, model=model,
                                                       step=steps,m_level=m_level, domain_name = domain_name)
        dmet.air_pressure_at_sea_level /= 100

        # prepare plot
        x,y = np.meshgrid(dmet.x, dmet.y)

        nx, ny = x.shape
        mask = (
          (x[:-1, :-1] > 1e20) |
          (x[1:, :-1] > 1e20) |
          (x[:-1, 1:] > 1e20) |
          (x[1:, 1:] > 1e20) |
          (x[:-1, :-1] > 1e20) |
          (x[1:, :-1] > 1e20) |
          (x[:-1, 1:] > 1e20) |
          (x[1:, 1:] > 1e20)
        )
         
        #print(OUTPUTPATH)
        #print(dt)
        # plot map
        lon0 = dmet.longitude_of_central_meridian_projection_lambert
        lat0 = dmet.latitude_of_projection_origin_projection_lambert
        parallels = dmet.standard_parallel_projection_lambert

        globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
        crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0,
                                    standard_parallels=parallels,globe=globe)
        #ax1 = plt.subplot(projection=crs)

        for tim in np.arange(np.min(steps), np.max(steps)+1, 1):

            # determine if image should be created for this time step
            stepok=False
            if tim<25:
                stepok=True
            elif (tim<=48) and ((tim % 3) == 0):
                stepok=True
            elif (tim<=66) and ((tim % 6) == 0):
                stepok=True
            if stepok==True:

                tidx = tim - np.min(steps)

                # calculate IVT
                IVT = wind_speed(dmet.x_wind_ml[tidx,:,:,:],dmet.y_wind_ml[tidx,:,:,:])
                #print('VEL')
                #print(np.shape(VEL))
                #print(VEL[:,100,100])
                #print(np.shape(VEL))
                IVT = IVT * dmet.specific_humidity_ml[tidx,:,:,:].squeeze()
                #print('IVT')
                #print(IVT[:,100,100])
                #p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
                # use an average dp
                dp=np.insert(np.diff(np.mean(dmet.surface_air_pressure[tidx,0,:,:].squeeze()) * dmet.b + dmet.ap),0,0)
                # calculate multiplication on axis without copying data
                shape = np.swapaxes(IVT, IVT.ndim-1, 0).shape
                B_brc = np.broadcast_to(dp, shape)
                # Swap back the axes. As before, this only changes our "point of view".
                B_brc = np.swapaxes(B_brc, IVT.ndim-1, 0)
                IVT = np.sum(IVT * B_brc,axis=0) / 9.81
                #print('IVT')
                #print(np.shape(IVT))
                #print(IVT[100,100])

                # calculate IWV
                IWV=dmet.specific_humidity_ml[tidx,:,:,:]
                #print(IWV[tidx,100,100])
                #p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
                dp=np.insert(np.diff(np.mean(dmet.surface_air_pressure[tidx,0,:,:].squeeze()) * dmet.b + dmet.ap),0,0)
                shape = np.swapaxes(IWV, IWV.ndim-1, 0).shape
                B_brc = np.broadcast_to(dp, shape)
                # Swap back the axes. As before, this only changes our "point of view".
                B_brc = np.swapaxes(B_brc, IWV.ndim-1, 0)
                IWV = np.sum(IWV * B_brc,axis=0) / 9.81
                #print('IWV')
                #print(np.shape(IWV))
                #print(IWV[100,100])

                fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9),subplot_kw={'projection': crs})
                ZS = dmet.surface_geopotential[tidx, 0, :, :]
                MSLP = np.where(ZS < 3000, dmet.air_pressure_at_sea_level[tidx, 0, :, :],
                                np.NaN).squeeze()
                CC=ax1.contourf(x,y,IVT.squeeze(),levels=[5,10,15,20,25,30,40,50,60,70,80,90,100,120,150,200],
                                cmap='tab20c',vmin=1,vmax=200,zorder=2,extend='both') 
                # MSLP
                # MSLP with contour labels every 10 hPa
                C_P = ax1.contour(dmet.x, dmet.y, MSLP, zorder=4, alpha=1.0,
                                  levels=np.arange(960, 1050, 1),
                                  colors='white', linewidths=0.5)
                C_P = ax1.contour(dmet.x, dmet.y, MSLP, zorder=5, alpha=1.0,
                                  levels=np.arange(960, 1050, 10),
                                  colors='white', linewidths=1.0, label="MSLP (hPa)")
                ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)
                ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),zorder=6,facecolor="none",edgecolor="k") 
                if domain_name != model and data_domain !=None: #weird bug.. cuts off when sees no data value
                     ax1.set_extent(data_domain.lonlat)
                ax1.text(0, 1, "{0}_IVT_{1}+{2:02d}".format(model, dt, tim), ha='left', va='bottom', \
                                       transform=ax1.transAxes, color='black')
                print("filename: "+make_modelrun_folder +"/{0}_{1}_{2}_{3}+{4:02d}.png".format(model, domain_name, "IVT", dt, tim))
                grid = True
                if grid:
                     nicegrid(ax=ax1)

                add_ISLAS_overlays(ax1)

                legend = True
                if legend:
                    
                    ax_cb = adjustable_colorbar_cax(fig1, ax1)

                    plt.colorbar(CC,cax = ax_cb, fraction=0.046, pad=0.01, aspect=25,
                                 label=r"IVT (kg m-1 s-1)")

                    proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray",zorder=7)]
                    # proxy.extend(proxy1)
                    # legend's location fixed, otherwise it takes very long to find optimal spot
                    lg = ax1.legend(proxy, ["MSLP [hPa]"],loc='upper left')
                    frame = lg.get_frame()
                    frame.set_facecolor('white')
                    frame.set_alpha(0.8)
                    proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray",zorder=7)]

                fig1.savefig(make_modelrun_folder +"/{0}_{1}_{2}_{3}+{4:02d}.png".format(model, domain_name, "IVT", dt, tim), bbox_inches="tight", dpi=200)
                ax1.cla()
                plt.clf()
                plt.close(fig1)

                # now plot IWV
                fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9),subplot_kw={'projection': crs})

                CC=ax1.contourf(x,y,IWV.squeeze(),levels=[1.0,2.0,3.0,4.0,5,6,7,8,9,10,12.5,15],
                                colors=('#FFFFFF','#ffffd9','#e0f3b2','#97d6b9','#41b6c4',
                                        '#1f80b8','#24429b','#081d58','#da30da','#a520a5','#600060'),vmin=1,
                                vmax=15,zorder=2,extend='both') 
                # MSLP
                # MSLP with contour labels every 10 hPa
                C_P = ax1.contour(dmet.x, dmet.y, MSLP, zorder=4, alpha=1.0,
                                  levels=np.arange(960, 1050, 1),
                                  colors='white', linewidths=0.5)
                C_P = ax1.contour(dmet.x, dmet.y, MSLP, zorder=5, alpha=1.0,
                                  levels=np.arange(960, 1050, 10),
                                  colors='white', linewidths=1.0, label="MSLP (hPa)")
                ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)
                ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),zorder=6,facecolor="none",edgecolor="k") 
                if domain_name != model and data_domain !=None: #weird bug.. cuts off when sees no data value
                     ax1.set_extent(data_domain.lonlat)
                ax1.text(0, 1, "{0}_IWV_{1}+{2:02d}".format(model, dt, tim), ha='left', va='bottom', \
                                       transform=ax1.transAxes, color='black')
                print("filename: "+make_modelrun_folder +"/{0}_{1}_{2}_{3}+{4:02d}.png".format(model, domain_name, "IWV", dt, tim))
                grid = True
                if grid:
                     nicegrid(ax=ax1)

                add_ISLAS_overlays(ax1)

                legend = True
                if legend:
                    
                    ax_cb = adjustable_colorbar_cax(fig1, ax1)

                    plt.colorbar(CC,cax = ax_cb, fraction=0.046, pad=0.01, aspect=25,
                                 label=r"IWV (kg m-2)")

                    proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray",zorder=7)]
                    # proxy.extend(proxy1)
                    # legend's location fixed, otherwise it takes very long to find optimal spot
                    lg = ax1.legend(proxy, ["MSLP [hPa]"],loc='upper left')
                    frame = lg.get_frame()
                    frame.set_facecolor('white')
                    frame.set_alpha(0.8)
                    proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray",zorder=7)]

                fig1.savefig(make_modelrun_folder +"/{0}_{1}_{2}_{3}+{4:02d}.png".format(model, domain_name, "IWV", dt, tim), bbox_inches="tight", dpi=200)
                ax1.cla()
                plt.clf()
                plt.close(fig1)
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
  parser.add_argument("--m_level", default=[0,64], nargs="+", type=int, help="summing over model levels --m_level 30 64 gives lowest 35 model levels")
  parser.add_argument("--model",default="MEPS", help="MEPS or AromeArctic")
  parser.add_argument("--domain_name", default=None, help="see domain.py", type = none_or_str)
  parser.add_argument("--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--legend", default=False, help="Display legend")
  parser.add_argument("--grid", default=True, help="Display legend")
  parser.add_argument("--info", default=False, help="Display info")
  parser.add_argument("--id", default=None, help="Display legend", type=str)
  parser.add_argument("--outpath", default=None, help="Display legend", type=str)
  args = parser.parse_args()

  #CHUNCK SIZE TO BIG
  s  = np.arange(np.min(args.steps),np.max(args.steps)+1)
  cn = np.int(len(s) // 4)
  if cn == 0:  # length of 6 not exceeded
      IVT_IWV(datetime=args.datetime, steps = [np.min(args.steps), np.max(args.steps)], model = args.model,
                 m_level=args.m_level,domain_name = args.domain_name, domain_lonlat=args.domain_lonlat,
                 legend = args.legend,info = args.info,grid=args.grid,runid =args.id, outpath=args.outpath)
  else: # lenght of 6 is exceeded, split in chunks, set by cn+1
      print(f"\n####### request exceeds 6 timesteps, will be chunked to smaller bits due to request limit ##########")
      chunks = np.array_split(s,cn+1)
      for c in chunks:
          IVT_IWV(datetime=args.datetime, steps = [np.min(c), np.max(c)], model = args.model,
                 m_level=args.m_level,domain_name = args.domain_name, domain_lonlat=args.domain_lonlat,
                 legend = args.legend,info = args.info,grid=args.grid,runid =args.id, outpath=args.outpath)
#
