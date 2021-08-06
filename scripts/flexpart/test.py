"""
Aina Johannessen.
2017/2018

Plot all trajectories


lon		longitude in deg
lat		latitude in deg
p		pressure in hPa
Q		specific humidity in g/kg
LWC	liquid water content in mg/kg
IWC		ice water content in mg/kg
RWC	rain water content in mg/kg
SWC	snow water content in mg/kg
RH		relative humidity in %
T		temperature in degC
TH		potential temperature in K
PV		potential vorticity in pvu
"""

import matplotlib  # Used for further calling

from datetime import datetime, timedelta

# from mpl_toolkits.basemap import Basemap
# from dypy.plotting import Mapfigure
# from lagranto import LagrantoRun
# from lagranto import Tra
import cartopy.crs as ccrs

# import cartopy.feature as cfeature
from lagranto.plotting import plot_trajs
import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
# import warnings
from lagranto import Tra


# ---GRAB data------------------


def set_background():
    # ------------Atlantic-----------------
    # m = Mapfigure(basemap = Basemap( llcrnrlon = -90., llcrnrlat = 10., urcrnrlon = 50., urcrnrlat=70., resolution = 'l', area_thresh = 10000., projection = 'merc' ))
    m = Mapfigure()

    # m = Basemap( llcrnrlon = -90., llcrnrlat = 10., urcrnrlon = 50., urcrnrlat=70.,\
    #           resolution = 'l', area_thresh = 10000., projection = 'merc' )
    # ------------Norway-------------------
    # m = Basemap( llcrnrlon = -30., llcrnrlat = 40., urcrnrlon = 40., urcrnrlat=70.,\
    #           resolution = 'h', area_thresh = 10000., projection = 'merc' )

    m.drawcoastlines(
        linewidth=0.5, linestyle="solid", color="k", zorder=5
    )  # [ 75./255., 75/255., 75/255. ] )
    m.drawmapboundary()  # fill_color='aqua')
    m.fillcontinents(
        color=[75.0 / 255.0, 75 / 255.0, 75 / 255.0], zorder=1
    )  # 'coral',lake_color='aqua')

    # --------draw parallels------
    circles = np.arange(-90.0, 90.0 + 30, 10.0)  # delat = 10.
    m.drawparallels(
        circles,
        color=[55.0 / 255.0, 55 / 255.0, 55 / 255.0],
        labels=[1, 0, 0, 0],
        linewidth=0.1,
        fontsize=7,
    )

    # --------draw meridians-----
    meridians = np.arange(0.0, 360, 10.0)  # delon = 10.
    m.drawmeridians(
        meridians,
        color=[55.0 / 255.0, 55.0 / 255.0, 55.0 / 255.0],
        labels=[0, 0, 0, 1],
        linewidth=0.1,
        fontsize=7,
    )
    return m


def plot_with_cartopy(rainout_trajs):
    print("--------------------------------")
    print("Plotting starts")
    crs = ccrs.Stereographic(
        central_longitude=-10, central_latitude=65, true_scale_latitude=65
    )

    fig = plt.figure(2)
    ax = plt.axes(projection=crs)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent([-90, 20, 20, 80])  # [lon0,lon1, lat0, lat1]
    cax = plot_trajs(ax, rainout_trajs, "Q")  # color with Q
    # cax = plot_trajs( ax, upt_idx, "Q" ) #color with Q
    # an = plt.annotate(trajs["time"])
    # for i in range(len(x)):
    #    plt.annotate(labls[i], xy=(x[i,2], y[i,2]), rotation=rotn[i,2])
    cbar = fig.colorbar(cax)
    # fig.set_size_inches( 12.80, 7.15 )
    # fig.savefig("./teeestcart.png", dpi = 600 )
    # plt.close( )
    return fig


def plot_with_basemap(rainout_trajs):
    fig = plt.figure(1)
    # ax = plt.subplot(111)

    m = set_background()
    CS = m.plot_traj(rainout_trajs, "Q")
    return fig


def correct_for_basemap_trouble(trajs):
    """
    -----------------
    Note: expressions bellow:
    Basemap through the method dypy.plotting crreates traj,
    if a traj passes -180 or 180, a horisontal line corss the whole globes appear
    trying to connect them
    Solution: Find which trajectories passes through 180 limit, break them up
     by setting values on border =np.nan.
    If this creates more prob, might remove everything bbeyond 180
    -----------------
    """

    idx_Border = np.where(
        (trajs["lon"] < -160) | (trajs["lon"] > 160)
    )  # | (trajs["lon"]>175 ) ) #idx of traj crossing boundary
    # upidx_Border = np.where(trajs["lon"]>160 )
    idx_exlBorder = np.where(
        (np.min(trajs["lon"], axis=1) >= -160) & (np.max(trajs["lon"], axis=1) <= 160)
    )  # all traj excluding the ones passing -179

    # Gather all traj excluding the one that crosses the border.
    trajs_coorected = Tra()
    trajs_coorected.set_array(
        trajs[idx_exlBorder[0], :]
    )  # make a copy of full array only with index that fulfill creataria

    # Gather all traj that crosses the border
    traj_Border = Tra()
    traj_Border.set_array(trajs[idx_Border[0], :])  # =all traj crossing border
    for idx in range(0, len(idx_Border[0])):  # set traj.values = nan at border
        sh = traj_Border["p"][:, idx_Border[1][idx] :].shape
        a = np.full(sh, np.nan)
        for var in trajs_coorected.variables:
            if var != "time":
                traj_Border[var][:, idx_Border[1][idx] :] = a
    trajs_coorected.append(traj_Border)  # append all traj and border trajs

    # uptraj_Border = Tra()
    # uptraj_Border.set_array( trajs[ upidx_Border[0],:]) #=all traj crossing border
    # for idx in range(0,len(upidx_Border[0])): #set traj.values = nan at border
    #   sh = uptraj_Border["p"][:,upidx_Border[1][idx]:].shape
    #   a = np.full(sh, np.nan)
    #   for var in trajs_coorected.variables:
    #      if var !='time':
    #         uptraj_Border[var][:,upidx_Border[1][idx]:]=a
    # trajs_coorected.append(uptraj_Border) #append all traj and border trajs

    return trajs_coorected


def traj_under700(t, file_in, trajs_main):
    print("for loop start")
    trajs = trajs_main[:, :].copy()
    print("traj read")
    rainout_index = np.where(
        ((trajs["Q"][:, 1] - trajs["Q"][:, 0]) > 0.0)
        & (np.min(trajs["p"][:, 0:6], axis=1) < 700)
    )  # 7592

    rainout_trajs = Tra()
    pickingsec = rainout_index[0]  # [::241*10]
    rainout_trajs.set_array(
        trajs[pickingsec, :]
    )  # make a copy of full array only with index that fulfill creataria
    # --------------------------------------------------------------------
    return rainout_trajs


def traj(t, file_in, trajs_main):
    print("for loop start")
    trajs = trajs_main[:, :].copy()
    print("traj read")
    rainout_index = np.where(
        ((trajs["Q"][:, 1] - trajs["Q"][:, 0]) > 0.0)
    )  # &  (np.min(trajs["lat"], axis=1) < 35 ) & (np.min(trajs["p"][:,0:6], axis =1)  < 700 )) #7592

    rainout_trajs = Tra()
    pickingsec = rainout_index[0]  # [::241*10]
    rainout_trajs.set_array(
        trajs[pickingsec, :]
    )  # make a copy of full array only with index that fulfill creataria
    # --------------------------------------------------------------------
    return rainout_trajs


def main():
    wrk_path = "/Users/ainajoh/Data/ISLAS/flexpart/test/"
    # path = wrk_path + "orig_traj/Walp_bigbox/"
    path_netcdf = wrk_path + "sel_traj/rainout_sources/1_all/netcdf/"
    times = ["20190210_00"]

    path_netcdf = "/Users/ainajoh/Data/ISLAS/flexpart/test/netcdf/"

    for t in times:  # lsl20190210_00
        filein = wrk_path + "lsl" + t
        fileout_all = path_netcdf + "all_" + str(t) + ".nc"
        # fileout_moist = path_netcdf + "moist_" + str(t) + '.nc'
        # fileout_moist_bel700 = path_netcdf + "moist_bel700" + str(t) + '.nc'

        print("Start reading main orig file")
        trajs_main = Tra()
        trajs_main.load_ascii(filein)
        # trajs_main = correct_for_basemap_trouble(trajs_main)
        trajs_main.write_netcdf(fileout_all)  # WRIRE
        print("NUMBER OF TRAJS!")
        print(np.shape(trajs_main))
        plot_with_cartopy(trajs_main)
        plt.show()
        # sel_trajs = traj(t, filein, trajs_main)
        # print("basemap correct starts")
        # sel_trajs = correct_for_basemap_trouble(sel_trajs)
        # sel_trajs.write_netcdf(fileout_moist)
        # print("NUMBER OF MOISTSELTRAJS!")
        # print(np.shape(sel_trajs))

        # sel_trajs = traj_under700(t, filein, trajs_main)
        # sel_trajs.write_netcdf(fileout_moist_bel700)
        # print("NUMBER OF MOISTSELTRAJS!")
        # print(np.shape(sel_trajs))

    # corr_trajs = correct_for_basemap_trouble(sel_trajs)
    # fig = plot_with_basemap(corr_trajs)
    # plot_with_cartopy(sel_trajs)
    # fig.set_size_inches( 12.80, 7.15 )
    # fig.savefig(event+times[0]+"<35.png", dpi = 600 )

    # plt.show()


# warnings.filterwarnings("ignore",category=matplotlib.mplDeprecation)

main()
