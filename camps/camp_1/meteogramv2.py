from domain import *
from netCDF4 import Dataset
from get_data import *
import pandas as pd
import os
import cartopy.crs as ccrs                      #For setting up a map

#print (os.getcwd())
path = os.path.abspath("islas/camps/camp_1")
map_domain = DOMAIN()
map_domain.KingsBay_Z0()

data_domain = DOMAIN()
data_domain.KingsBay_Z1()

param_ML = ["air_temperature_ml","x_wind_ml","y_wind_ml","specific_humidity_ml"]
param_SFC = ["atmosphere_boundary_layer_thickness","air_temperature_2m", "surface_air_pressure","relative_humidity_2m",\
             "x_wind_10m", "y_wind_10m","x_wind_gust_10m", "y_wind_gust_10m",\
             "integral_of_surface_downward_sensible_heat_flux_wrt_time",\
             "integral_of_surface_downward_latent_heat_flux_wrt_time","land_area_fraction"]
#sites = pd.read_csv(path + "/sites.csv", sep=";", header=0, index_col=0)
#sites.loc["OldPier"]
#d1 = DATA(data_domain=data_domain, param_ML=param_ML, param_SFC = param_SFC)
d1 = DATA(data_domain=data_domain)
d1.retrieve()
print(d1.latitude)
#print(d1.param) = list all param name defined.



def background_map( ):
    projection = ccrs.UTM( 33 ) #Map projection you want
    crs_latlon = ccrs.PlateCarree() #Coordinates used to plot points on map

    fig = plt.subplots( figsize=(10, 9) )

    ax = plt.axes( projection=projection )
    ax.set_extent( latlon_mapdomain )
    #LOAD SHAPEFILE FOR COAST and HEIGHT CONTOURS
    shp = shapereader.Reader('/Users/ainajoh/Data/ISAS/shape_files/NP_S100_SHP/S100_Land_f.shp')
    shpkvote=shapereader.Reader('/Users/ainajoh/Data/ISAS/shape_files/NP_S100_SHP/S100_Koter_l.shp')
    #PLOT GRIDLINES
    gl = ax.gridlines(crs=crs_latlon, draw_labels=False, linewidth=1, color='gray', alpha=0.3, linestyle='--')
    #########################################
    #PLOT COASTLINE AND HIGHT CONTOURS
    #########################################
    #Coastline
    ax.add_geometries( shp.geometries(), projection, facecolor='gray', edgecolor='black', zorder=2 )
    #Heightcontours
    ax.add_geometries( shpkvote.geometries(), projection, facecolor="None", edgecolor='white', alpha = 0.1,  zorder=2 )


def plot_site():
    #PLOT COORDINATES FOR OLDPIER
    plt.plot(latlon_old_pier[0], latlon_old_pier[1], marker='o', markersize=5.0, markeredgewidth=2.5,
                     markerfacecolor='blue', markeredgecolor='blue', zorder=6, transform=crs_latlon)

    #PLOT GRIDPOINT OF INTEREST
    points = plt.plot(lons_gridPointDomain,lats_gridPointDomain, marker='.', markersize=5.0, markeredgewidth=4,
                     markerfacecolor='black', markeredgecolor='black', zorder=6, transform=crs_latlon, linestyle = 'None')

    return projection, crs_latlon
