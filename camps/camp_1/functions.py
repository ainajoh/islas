from netCDF4 import Dataset                     #For reading netcdf files.
import cartopy.crs as ccrs                      #For setting up a map
from cartopy.io import shapereader              #For reading shapefiles containg high-resolution coastline.
import matplotlib as mpl
import matplotlib.pyplot as plt                 #For basic plotting in python
import matplotlib.cm as cm                      #For colors on map
import datetime as dt
import matplotlib.dates as mdates
import numpy as np


def define_domain( latlon_mapdomain = [11, 14, 78.8, 79.2], adjusted_map2datadomain = [0.7, -1.1, 0.08, 0.2]):

    if latlon_mapdomain == [11, 14, 78.8, 79.2] or adjusted_map2datadomain == [0.7, -1.1, 0.08, 0.2]:
        print("using prefix variables")
        idx = np.array(
            [517, 517, 518, 518, 518, 518, 518, 519, 519, 519, 519, 519, 519, 520, 520, 520, 520, 520, 520, 520,
             520, 520, 521, 521, 521, 521, 521, 521, 521, 521, 521, 522, 522, 522, 522, 522, 522, 522, 522, 522,
             523, 523, 523, 523, 523, 523, 524, 524, 524, 524, 524, 525, 525, 525],
            [183, 184, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 187, 181, 182, 183, 184, 185, 186, 187,
             188, 189, 182, 183, 184, 185, 186, 187, 188, 189, 190, 183, 184, 185, 186, 187, 188, 189, 190, 191,
             185, 186, 187, 188, 189, 190, 186, 187, 188, 189, 190, 187, 188, 189])
        latlon_datadomain = [a + b for a, b in zip(latlon_mapdomain, adjusted_map2datadomain)]
    else:
        print("making new data_domain")
        #import data_extract_domain as new_domain
        idx, latlon_datadomain = [0,0]#new_domain.data_domain(latlon_mapdomain, adjusted_map2datadomain)

    return idx, latlon_datadomain

class meteogram():
    def __init__(self):
        #self.latlon_mapdomain = [11, 14, 78.8, 79.2] # Kingsbay area, domain of interest
        self.latlon_mapdomain = [11, 14, 78.8, 79.2] # Kingsbay area, domain of interest
        adjusted_map2datadomain = [0.7, -1.1, 0.08, 0.2]
        self.latlon_old_pier = [11.91929, 78.93030] # end of old pier
        self.idx =None
        self.latlon_datadomain = None

        define_domain( self.latlon_mapdomain, adjusted_map2datadomain)

        jindx = idx[0]#np.array([517, 517, 518, 518, 518, 518, 518, 519, 519, 519, 519, 519, 519,520, 520, 520, 520, 520, 520, 520, 520, 520, 521, 521, 521, 521,521, 521, 521, 521, 521, 522, 522, 522, 522, 522, 522, 522, 522, 522, 523, 523, 523, 523, 523, 523, 524, 524, 524, 524, 524, 525, 525, 525])
        iindx = idx[1]#np.array([183, 184, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 187,181, 182, 183, 184, 185, 186, 187, 188, 189, 182, 183, 184, 185, 186, 187, 188, 189, 190, 183, 184, 185, 186, 187, 188, 189, 190, 191, 185, 186, 187, 188, 189, 190, 186, 187, 188, 189, 190, 187, 188, 189])


        #self.url = f"https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_full_2_5km_latest.nc?latitude[{jindx}][{iindx}],longitude[{jindx}][{iindx}]"
        self.url = f"https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_full_2_5km_latest.nc?"+\
                    f"latitude[{jindx.min()}:1:{jindx.max()}][{iindx.min()}:1:{iindx.max()}],"+ \
                    f"longitude[{jindx.min()}:1:{jindx.max()}][{iindx.min()}:1:{iindx.max()}]"


        dataset = Dataset(self.url)
        self.lon = dataset.variables["longitude"][:]
        self.lat = dataset.variables["latitude"][:]
        # lon/lat points for complete region
        #lon_index = ((dataset.variables["longitude"][:] > self.latlon_focus_area[0]) & (dataset.variables["longitude"][:] < self.latlon_focus_area[1]))
        #lat_index = ((dataset.variables["latitude"][:] > self.latlon_focus_area[2]) & (dataset.variables["latitude"][:] < self.latlon_focus_area[3]))

        def define_domain(self, latlon_mapdomain=[11, 14, 78.8, 79.2], adjusted_map2datadomain=[0.7, -1.1, 0.08, 0.2]):

            if latlon_mapdomain == [11, 14, 78.8, 79.2] or adjusted_map2datadomain == [0.7, -1.1, 0.08, 0.2]:
                print("using prefix variables")
                idx = np.array(
                    [517, 517, 518, 518, 518, 518, 518, 519, 519, 519, 519, 519, 519, 520, 520, 520, 520, 520, 520, 520,
                     520, 520, 521, 521, 521, 521, 521, 521, 521, 521, 521, 522, 522, 522, 522, 522, 522, 522, 522, 522,
                     523, 523, 523, 523, 523, 523, 524, 524, 524, 524, 524, 525, 525, 525],
                    [183, 184, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 187, 181, 182, 183, 184, 185, 186, 187,
                     188, 189, 182, 183, 184, 185, 186, 187, 188, 189, 190, 183, 184, 185, 186, 187, 188, 189, 190, 191,
                     185, 186, 187, 188, 189, 190, 186, 187, 188, 189, 190, 187, 188, 189])
                latlon_datadomain = [a + b for a, b in zip(latlon_mapdomain, adjusted_map2datadomain)]
            else:
                print("making new data_domain")
                # import data_extract_domain as new_domain
                idx, latlon_datadomain = [0, 0]  # new_domain.data_domain(latlon_mapdomain, adjusted_map2datadomain)

            return idx, latlon_datadomain



            #from data_extract_domain import data_domain(latlon_mapdomain=[11, 14, 78.8, 79.2], a=0.7, b=-1.1, c=0.08, d=-0.2):


        #dataset


        # lon/lat points within focus area
        #indx = np.where([(lon > self.lonmin) & (lon < self.lonmax) & (lat >= self.latmin) & (lat <= self.latmax)])
        #dataset[indx]

        #lon_focus = lon[indx[1], indx[2]]
        #lat_focus = lat[indx[1], indx[2]]
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
    #shpkvote=shapereader.Reader('/Users/ainajoh/Data/ISAS/shape_files/NP_S100_SHP/S100_Koter_l.shp')
    #PLOT GRIDLINES
    gl = ax.gridlines(crs=crs_latlon, draw_labels=False, linewidth=1, color='gray', alpha=0.3, linestyle='--')
    #########################################
    #PLOT COASTLINE AND HIGHT CONTOURS
    #########################################
    #Coastline
    ax.add_geometries( shp.geometries(), projection, facecolor='gray', edgecolor='black', zorder=2 )
    #Heightcontours
    #ax.add_geometries( shpkvote.geometries(), projection, facecolor="None", edgecolor='white', alpha = 0.1,  zorder=2 )
    #######################################
    #PLOT COORDINATES FOR OLDPIER
    plt.plot(latlon_old_pier[0], latlon_old_pier[1], marker='o', markersize=5.0, markeredgewidth=2.5,
                     markerfacecolor='blue', markeredgecolor='blue', zorder=6, transform=crs_latlon)
    #
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
    plt.show()
    #return projection, crs_latlon


a = meteogram()
background_map(a.latlon_old_pier, a.latlon_mapdomain, a.lon, a.lat)

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

