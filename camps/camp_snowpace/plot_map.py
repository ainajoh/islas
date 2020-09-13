from loclib.config import *  # require netcdf4
from loclib.domain import *  # require netcdf4
from loclib.get_data import *
from loclib.calculation import *
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.basemap import Basemap

lt = 2  # lead time.
modelruntime = "2020030800"
#MAP DOMAIN SMALLER THAN DATA DOMAIN DUE TO BORDERS
map_domain = DOMAIN()
map_domain.South_Norway()
lonlat = np.array(map_domain.lonlat)

map_data_domain = DOMAIN()
map_data_domain.South_Norway()
##########################

param_SFC = ["x_wind_10m", "y_wind_10m",
             "air_temperature_2m","air_pressure_at_sea_level" ]  # ["air_temperature_2m", "surface_air_pressure", "air_pressure_at_sea_level",

dmap = DATA( data_domain=map_data_domain, model="MEPS", param_SFC=param_SFC, fctime=[0, lt], modelrun=modelruntime)
dmap.retrieve()
time_normal = timestamp2utc(dmap.time)
modelrun = timestamp2utc([dmap.forecast_reference_time])

def background_map(lonlat):
    map = Basemap(llcrnrlon=lonlat[0], llcrnrlat=lonlat[2], urcrnrlon=lonlat[1], urcrnrlat=lonlat[3],
                  resolution='f', projection="tmerc", lon_0=15., lat_0=42.,
                  area_thresh=0.0001)  # "epsg=5973,

    map.drawcoastlines()
    #map.readshapefile('./bin/shapefiles/svalbard/S100_Land_f_WGS84', 'S100_Land_f_WGS84',
    #                  zorder=1000, linewidth=2)

    return map

def mapwithdata(dirName_b2, figname_b2):
    figm2, ax = plt.subplots(figsize=(10, 9))
    map = background_map(lonlat)
    for t in range(0, np.shape(dmap.time)[0]):
        figm2, ax = plt.subplots(figsize=(10, 9))
        map = background_map(lonlat)

        cmap = cm.get_cmap('twilight_shifted')
        x, y = map(dmap.longitude, dmap.latitude)
        CFW = plt.pcolormesh(x, y, dmap.air_temperature_2m[t, 0, :, :] - 273.15, zorder=10, alpha=0.9,
                             cmap=cmap, vmin = -30, vmax = 0)
        plt.Rectangle((0.88, 0.84), 0.12, 0.15, fc=[1, 1, 1, 0.7])
        # ticks = np.linspace(np.min(dmap.air_temperature_2m), np.max(dmap.air_temperature_2m), 4)
        cbar = plt.colorbar(CFW, extend="both", format='%.0f', fraction=0.02, pad=0.01, ax = None)

        cbar.set_label('2m Temp. [C]')

        map.contour(x,y, dmap.air_pressure_at_sea_level[t, 0, :, :]/100, colors="k",zorder=10)
        wspeed = np.sqrt(dmap.x_wind_10m[t, 0, :, :] ** 2 + dmap.y_wind_10m[t, 0, :, :] ** 2)
        plt.barbs(x, y, dmap.x_wind_10m[t, 0, :, :], dmap.y_wind_10m[t, 0, :, :], color="black", zorder=20)
        figname_b2_1 = figname_b2 + "+" + str(t)

        plt.savefig(dirName_b2 + figname_b2 + ".png")
        plt.close()

def setup_directory():
    projectpath = "../../output/"
    figname = "fc_" + modelrun[0].strftime('%Y%m%d%H')
    # dirName = projectpath + "result/" + modelrun[0].strftime('%Y/%m/%d/%H/')
    dirName = projectpath + "result/" + "fc_" + modelrun[0].strftime('%Y%m%d/')

    dirName_b1 = dirName + "met/"
    figname_b1 = "met_" + figname

    dirName_b2 = dirName + "map/"
    figname_b2 = "dmap_" + figname

    if not os.path.exists(dirName_b1):
        os.makedirs(dirName_b1)
        print("Directory ", dirName_b1, " Created ")
    else:
        print("Directory ", dirName_b1, " already exists")

    if not os.path.exists(dirName_b2):
        os.makedirs(dirName_b2)
        print("Directory ", dirName_b2, " Created ")
    else:
        print("Directory ", dirName_b2, " already exists")
    return dirName_b1, dirName_b2, figname_b1, figname_b2

def Arctic(dirName_b2,figname_b2):
    map_domain = DOMAIN()
    map_domain.South_Norway()
    lonlat = np.array(map_domain.lonlat)

    map_data_domain = DOMAIN()
    map_data_domain.South_Norway()
    param_SFC = ["x_wind_10m", "y_wind_10m",
                 "air_temperature_2m", "air_pressure_at_sea_level"]  # ["air_temperature_2m", "surface_air_pressure", "air_pressure_at_sea_level",

    dmap = DATA(data_domain=map_data_domain, param_SFC=param_SFC, fctime=[0, lt], modelrun=modelruntime)
    dmap.retrieve()
    print("calc uv")
    u, v = xwind2uwind(dmap.x_wind_10m, dmap.y_wind_10m, dmap.alpha)
    print("DONE CALC UV")

    time_normal = timestamp2utc(dmap.time)
    modelrun = timestamp2utc([dmap.forecast_reference_time])

    for t in range(0, np.shape(dmap.time)[0]):
        figm2, ax = plt.subplots(figsize=(7, 9))
        map = Basemap(llcrnrlon=lonlat[0], llcrnrlat=lonlat[2], urcrnrlon=lonlat[1], urcrnrlat=lonlat[3],
                      resolution='f', projection="tmerc", lon_0=15., lat_0=42.,
                      area_thresh=0.0001)  # "epsg=5973,

        map.drawcoastlines(linewidth=2.0, color='gray', ax=ax, zorder=1000)

        cmap = cm.get_cmap('twilight_shifted')
        x, y = map(dmap.longitude, dmap.latitude)

        n=5
        yy = np.arange(0, y.shape[0], 10)  # skips over every 100th value 100 = inbetween
        xx = np.arange(0, x.shape[1], 10)  # skips over every 100th value
        points = np.meshgrid(yy, xx)
        us = u[t,0,:,:]
        vs = v[t,0,:,:]
        ur, vr = map.rotate_vector(us, vs, dmap.longitude, dmap.latitude)

        clr_levels = np.arange(-20,15,3)
        CF=map.contourf( x, y, dmap.air_temperature_2m[t,0,:,:]-273.15, cmap = cmap,zorder=1,levels=clr_levels,extend="both")
        plt.Rectangle((0.88, 0.84), 0.12, 0.15, fc=[1, 1, 1, 0.7])
        cbara = plt.colorbar(CF, extend="both", format='%.0f', fraction=0.02, pad=0.01, ax=None)
        cbara.set_label('2m Temp. [C]')

        map.barbs(x[points], y[points], ur[points], vr[points], color="k", zorder=2)

        lvl = np.arange(950,1050,1)
        CL = map.contour(x,y, dmap.air_pressure_at_sea_level[t, 0, :, :]/100,levels=lvl, colors="k",zorder=10)
        plt.savefig(dirName_b2 + figname_b2 + "LOC[S].png")
        plt.close()

def main():
    dirName_b1, dirName_b2, figname_b1, figname_b2 = setup_directory()
    Arctic(dirName_b2,figname_b2)

main()