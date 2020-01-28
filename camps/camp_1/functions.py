from netCDF4 import Dataset                     #For reading netcdf files.
import cartopy.crs as ccrs                      #For setting up a map
from cartopy.io import shapereader              #For reading shapefiles containg high-resolution coastline.
import matplotlib as mpl
import matplotlib.pyplot as plt                 #For basic plotting in python
import matplotlib.cm as cm                      #For colors on map
import datetime as dt
import matplotlib.dates as mdates
import numpy as np



class meteogram():
    def __init__(self):
        self.latlon_mapdomain = [11, 14, 78.8, 79.2] # Kingsbay area, domain of interest
        self.latlon_old_pier = [11.91929, 78.93030] # end of old pier
        self.latlon_focus_area = [a + b for a, b in zip(self.latlon_mapdomain, [0.7, -1.1, 0.08, -0.2])]

        # download data
        self.url = f"https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_vc_2_5km_latest.nc?latitude[{self.latlon_mapdomain[0]}:0:0][{self.latlon_mapdomain[1]}:0:0],longitude[{self.latlon_mapdomain[2]}:0:0][{self.latlon_mapdomain[3]}:0:0]"
        #url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_full_2_5km_latest.nc"
        dataset = Dataset(url)

        # lon/lat points for complete region
        lon_index = ((dataset.variables["longitude"][:] > self.latlon_focus_area[0]) & (dataset.variables["longitude"][:] < self.latlon_focus_area[1]))
        lat_index = ((dataset.variables["latitude"][:] > self.latlon_focus_area[2]) & (dataset.variables["latitude"][:] < self.latlon_focus_area[3]))

        dataset


        # lon/lat points within focus area
        indx = np.where([(lon > self.lonmin) & (lon < self.lonmax) & (lat >= self.latmin) & (lat <= self.latmax)])
        dataset[indx]

        lon_focus = lon[indx[1], indx[2]]
        lat_focus = lat[indx[1], indx[2]]





        dataset.close()




def background_map(latlon_old_pier, latlon_mapdomain, lons_gridPointDomain, lats_gridPointDomain):
    """
    Make map. Using shapefile for coast for high enough resolution.
    """
    #########################################
    # SETUP PROJECTION AND MAP ELEMENTS
    #########################################
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
    #######################################
    #PLOT COORDINATES FOR OLDPIER
    plt.plot(latlon_old_pier[0], latlon_old_pier[1], marker='o', markersize=5.0, markeredgewidth=2.5,
                     markerfacecolor='blue', markeredgecolor='blue', zorder=6, transform=crs_latlon)

    #PLOT GRIDPOINT OF INTEREST
    points = plt.plot(lons_gridPointDomain,
                      lats_gridPointDomain,
                      marker='.',
                      markersize=5.0,
                      markeredgewidth=4,
                     markerfacecolor='black',
                      markeredgecolor='black',
                      zorder=6,
                      transform=crs_latlon,
                      linestyle = 'None')

    return projection, crs_latlon

def grid_point(lonp, latp, crs_latlon):
    """
    Visualise the gridpoint of interest for when plotting meteograms.
    """
    #color gridpoint red
    plt.plot(lonp, latp, marker='.', markersize=5.0, markeredgewidth=4,
                      markerfacecolor='red', markeredgecolor='red', zorder=7, transform=crs_latlon,
                      linestyle='None')
    plt.savefig("AROME_PLOT_lon"+str(lonp) + "__lat" + str(latp) + ".png")

    #color gridpoint black again
    plt.plot(lonp, latp, marker='.', markersize=5.0, markeredgewidth=4,
                      markerfacecolor='black', markeredgecolor='black', zorder=7, transform=crs_latlon,
                      linestyle='None')






def main():
    old_pier, latlon, londomain, latdomain, jindx, iindx = Svalbard()
    projection, crs_latlon = background_map(old_pier, latlon, londomain, latdomain)
    test = 0
    #meteogram_average(jindx, iindx )
    for i in range(0,np.shape(londomain)[0]):
        while test == 0:
            print(test)
            #grid_point(londomain[i], latdomain[i], crs_latlon)
            meteogram_vertical(jindx[i], iindx[i])
            meteogram(jindx[i], iindx[i])
            test += 1

