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
from add_overlays import *


def Cloud_base_top(datetime, steps, model, domain_name = None, domain_lonlat = None, legend=False, info = False, grid=True, runid=None, outpath=None):
    global OUTPUTPATH
    if outpath != None:
        OUTPUTPATH=outpath

    for dt in datetime: #modelrun at time..
        if runid !=None:
            make_modelrun_folder = setup_directory( OUTPUTPATH, "{0}-{1}".format(dt,runid) )
        else:
            make_modelrun_folder = setup_directory( OUTPUTPATH, "{0}".format(dt) )
    # can be replaced at some time with proper height at model levels, but works for now        
    H = [24122.6894480669, 20139.2203688489,17982.7817599549, 16441.7123200128,
    15221.9607620438, 14201.9513633491, 13318.7065659522, 12535.0423836784,
    11827.0150898454, 11178.2217936245, 10575.9136768674, 10010.4629764989,
    9476.39726730647, 8970.49319005479, 8490.10422494626, 8033.03285976169,
    7597.43079283063, 7181.72764002209, 6784.57860867911, 6404.82538606181,
    6041.46303718354, 5693.61312218488, 5360.50697368367, 5041.46826162131,
    4735.90067455394, 4443.27792224573, 4163.13322354697, 3895.05391218293,
    3638.67526925036, 3393.67546498291, 3159.77069480894, 2936.71247430545,
    2724.28467132991, 2522.30099074027, 2330.60301601882, 2149.05819142430,
    1977.55945557602, 1816.02297530686, 1664.38790901915, 1522.61641562609,
    1390.69217292080, 1268.36594816526, 1154.95528687548, 1049.75817760629,
    952.260196563843, 861.980320753114, 778.466725603312, 701.292884739207,
    630.053985133223, 564.363722589458, 503.851644277509, 448.161118360263,
    396.946085973573, 349.869544871297, 306.601457634038, 266.817025119099,
    230.194566908004, 196.413229972062, 165.151934080260, 136.086183243070,
    108.885366240509, 83.2097562375566,58.7032686584901, 34.9801888163106,
    11.6284723290378]
    for dt in datetime:
        date = dt[0:-2]
        hour = int(dt[-2:])
        all_param = ['cloud_base_altitude','cloud_top_altitude','air_pressure_at_sea_level','surface_geopotential','cloud_area_fraction_ml','high_type_cloud_area_fraction']
        dmet,data_domain,bad_param = checkget_data_handler(all_param=all_param, date=dt, model=model,
                                                           step=steps, domain_name=domain_name)
        
        dmet.air_pressure_at_sea_level /= 100
        # Cloud top, cloud bse
        # print(dmet.pressure) 
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

        # plot map
        lon0 = dmet.longitude_of_central_meridian_projection_lambert
        lat0 = dmet.latitude_of_projection_origin_projection_lambert
        parallels = dmet.standard_parallel_projection_lambert

        globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
        crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0,
                                    standard_parallels=parallels,globe=globe)
           
        for tim in np.arange(np.min(steps), np.max(steps)+1, 1):

            #ax1 = plt.subplot(projection=crs)
 
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
                ttt = tim #+ np.min(steps)
                tidx = tim - np.min(steps)
 
                ZS = dmet.surface_geopotential[tidx, 0, :, :]
                MSLP = np.where(ZS < 3000, dmet.air_pressure_at_sea_level[tidx, 0, :, :],
                                np.NaN).squeeze()
                
                CT   = dmet.cloud_top_altitude[tidx,0,:,:].copy()
                CB   = dmet.cloud_base_altitude[tidx,0,:,:].copy()
                CB[np.where(CB > 20000)] = np.nan # get rid of lage fill values
                # take three cloud level intervalls, to idicate by markers
                # level are 0-1000m, 1000-2000m, 2000-3000m 
                CB1 = CB.copy()
                CB2 = CB.copy()
                CB3 = CB.copy() 
                CB1[np.where(CB>1000)]        = np.nan
                CB2[np.where(~np.isnan(CB1))] = np.nan # no double counting
                CB2[np.where(CB>2000)]        = np.nan
                CB3[np.where(~np.isnan(CB1))] = np.nan # no double counting
                CB3[np.where(~np.isnan(CB2))] = np.nan # no double counting
                CB3[np.where(CB>3000)]        = np.nan
 
                HH = np.zeros([65,949,739])
                for i,z in enumerate(H):
                    HH[i,:,:] = z + dmet.surface_geopotential[tidx,0,:,:]/9.81
                
                caf3 = dmet.cloud_area_fraction_ml[tidx,:,:,:].copy()
                caf3 = caf3[14:65,:,:] # only the levels wie are interested in
                buf = np.zeros(caf3.shape)
                thresh = 0.43 # threshold for cloud ..kind of aligns with MET values
                buf[np.where(caf3>=thresh)] = 1  
                HH3 = HH[14:65,:,:]
                ### this can for sure be improved!
                CT_new = np.zeros([949,739])
                for z in range(HH3.shape[0]-1,0,-1):# go from lowest to topmost and overwrite values
                    xx,yy = np.where(buf[z,:,:]==1)
                    CT_new[xx,yy] = HH3[z,xx,yy]
                CT_new[np.where(CT_new==0)] = np.nan # filter out cloud free areas
                # plot 
                #CT2 = CT.copy()
                # set all cloud tops above 14000m to nan, choose 14000 to align more with LMH plot
                #CT2[np.where(CT2>14000)] = np.nan
                #data =  CT2[:nx - 1, :ny - 1].copy()
                data =  CT_new[:nx - 1, :ny - 1].copy()
                data[mask] = np.nan
                # making a contour for cirrus clouds, make a rough estimate of 0.5
                highC = dmet.high_type_cloud_area_fraction[tidx,0,:,:]
                highC[np.where(highC>0.5)] = 1
                highC[np.where(highC<=0.5)] = 0
                cmap = plt.cm.get_cmap('rainbow_r', 9)
                cmap.set_over('lightgrey')
                CCl   = ax1.pcolormesh(x, y,  data[:, :], cmap=cmap,
                                       vmin=0, vmax=9000,zorder=2)
                # indicate cloud base height by markers
                co = '#393939'
                skip = (slice(10, None, 20), slice(10, None, 20))
                xx = x.copy()
                yy = y.copy()
                xx[np.where(np.isnan(CB1))] = np.nan
                yy[np.where(np.isnan(CB1))] = np.nan
                sc1 = ax1.scatter(xx[skip], yy[skip], s=30, zorder=2, marker='o', linewidths=0.9,
                                  c=co, alpha=0.75,label='[0m, 1000 m]')
                xx = x.copy()
                yy = y.copy()
                xx[np.where(np.isnan(CB2))] = np.nan
                yy[np.where(np.isnan(CB2))] = np.nan
                sc2 = ax1.scatter(xx[skip], yy[skip], s=30, zorder=2, marker='o', linewidths=0.9,
                                  facecolors='none',edgecolors=co, alpha=0.75,label='[1000m, 2000m]')
                xx = x.copy()
                yy = y.copy()
                xx[np.where(np.isnan(CB3))] = np.nan
                yy[np.where(np.isnan(CB3))] = np.nan
                sc3 = ax1.scatter(xx[skip], yy[skip], s=30, zorder=2, marker='x', linewidths=0.9,
                       c=co, alpha=0.75,label='[2000m, 3000m]')
                # MSLP
                # MSLP with contour labels every 10 hPa, commented out due to cluttering
                # for now commeted out for readability
                #C_P = ax1.contour(dmet.x, dmet.y, MSLP, zorder=3, alpha=1.0,
                #                  levels=np.arange(960, 1050, 1),
                #                  colors='grey', linewidths=0.5)
                #C_P = ax1.contour(dmet.x, dmet.y, MSLP, zorder=3, alpha=1.0,
                #                  levels=np.arange(960, 1050, 10),
                #                  colors='grey', linewidths=1.0, label="MSLP (hPa)")
                #ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

                # instead plot a contour of high cloud cover for clarity of plot
                ax1.contour(x, y, highC, colors='grey',zorder=2)
                ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),zorder=7,facecolor="none",edgecolor="black") 
                # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
                if domain_name != model and data_domain !=None: #weird bug.. cuts off when sees no data value
                     ax1.set_extent(data_domain.lonlat)
                ax1.text(0, 1, "{0}_CB_CT_{1}_+{2:02d}".format(model, dt, ttt), ha='left', va='bottom', \
                                       transform=ax1.transAxes, color='black')
                print("filename: "+make_modelrun_folder +"/{0}_{1}_{2}_{3}+{4:02d}.png".format(model, domain_name, "CB_CT", dt, ttt))
                grid = True
                if grid:
                    nicegrid(ax=ax1)

                add_ISLAS_overlays(ax1,col='red')
 
                legend = True
                if legend:
                    ax_cb = adjustable_colorbar_cax(fig1, ax1)

                    plt.colorbar(CCl,cax = ax_cb, fraction=0.046, pad=0.01, aspect=25,
                                 label=r"cloud top height (m)",extend='max')
                    l1 = ax1.legend(loc='upper left')
                    frame = l1.get_frame()
                    frame.set_facecolor('white')
                    frame.set_alpha(1)
 
                fig1.savefig(make_modelrun_folder +"/{0}_{1}_{2}_{3}+{4:02d}.png".format(model, domain_name, "CB_CT", dt, ttt), bbox_inches="tight", dpi=200)
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
  parser.add_argument("--model",default="MEPS", help="MEPS or AromeArctic")
  parser.add_argument("--domain_name", default=None, help="see domain.py", type = none_or_str)
  parser.add_argument("--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--legend", default=False, help="Display legend")
  parser.add_argument("--grid", default=True, help="Display legend")
  parser.add_argument("--info", default=False, help="Display info")
  parser.add_argument("--id", default=None, help="Display legend", type=str)
  parser.add_argument("--outpath", default=None, help="Display legend", type=str)

  args = parser.parse_args()
  
  Cloud_base_top(datetime=args.datetime, steps = [np.min(args.steps), np.max(args.steps)], model = args.model,
                     domain_name = args.domain_name, domain_lonlat=args.domain_lonlat,
                     legend = args.legend,info = args.info,grid=args.grid, runid =args.id, outpath=args.outpath)


