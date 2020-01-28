#Todo: 0: Color boundary hight better.. White/Black: Be done with this by 15:30, and start with reg meteograms.
#       1: Make nice temp color with humidity cont vis enough. Decide if q or rh is best..
#       1.1 Height as y value as well
#      2:I want either temp background, humidity as contours.
#      #Or humidy as background, and pot temp as contours. Takes a while calc pot temp, so maybe skip this.
#      3: See if cloud cover is cool ? Maybe incoming radiation..?
#      4: Start with regular meteograms. Precip also. fluxes ground.
#      5: Harald wanted spagetti. So make a new map  that takes all grid in the bay using land fraction?
#           Ide would be; if land = true of point, save to variable land_points.
#           What about mointain vs ground level ? Is there a height for each grid point? remember its supposed to reopresnt average in gridcell..


#from islas.camps.camp_1.functions import test




from netCDF4 import Dataset                     #For reading netcdf files.
import cartopy.crs as ccrs                      #For setting up a map
from cartopy.io import shapereader              #For reading shapefiles containg high-resolution coastline.
import matplotlib as mpl
import matplotlib.pyplot as plt                 #For basic plotting in python
import matplotlib.cm as cm                      #For colors on map
import datetime as dt
import matplotlib.dates as mdates
import numpy as np


url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_full_2_5km_latest.nc"
print("Reads the data")
dataset = Dataset(url)
#########################################
#EXTRACT PARAMETERS
#########################################
print("Extract the data")
#x = dataset.variables["x"]
#y = dataset.variables["y"]
lon = dataset.variables["longitude"][:]
lat = dataset.variables["latitude"][:]
tmp_ml = dataset.variables["air_temperature_ml"] #(time, hybrid, y, x) (67, 65, 949, 739)
x_wind_ml = dataset.variables["x_wind_ml"]
y_wind_ml = dataset.variables["y_wind_ml"]
specific_humidity_ml = dataset.variables["specific_humidity_ml"]
# cloud_area_fraction_ml
BL_meter = dataset.variables["atmosphere_boundary_layer_thickness"][:]

#normal surface meteograms
air_temperature_2m = dataset.variables["air_temperature_2m"][:] #(time,height1,y,x)
relative_humidity_2m = dataset.variables["relative_humidity_2m"][:]
x_wind_10m = dataset.variables["x_wind_10m"][:]
y_wind_10m = dataset.variables["y_wind_10m"][:]
x_wind_gust_10m = dataset.variables["x_wind_gust_10m"][:]
y_wind_gust_10m = dataset.variables["y_wind_gust_10m"][:]

SH = dataset.variables["integral_of_surface_downward_sensible_heat_flux_wrt_time"][:]
LH = dataset.variables["integral_of_surface_downward_latent_heat_flux_wrt_time"][:]

land_area_fraction = dataset.variables["land_area_fraction"][:]
time = dataset.variables["time"][:] #67
hybrid = dataset.variables["hybrid"][:] #65
ap = dataset.variables["ap"][:] #65
b = dataset.variables["b"][:] #65
ps = dataset.variables["surface_air_pressure"][:] #surface pressure  Pa
#########################################
#CALCULATE VARIABLES
#########################################
print("Calculate new variables")
#Convert time to readable format from seconds since 1970-01-01
time_normal = [dt.datetime.utcfromtimestamp(x) for x in time]
#Calculste boundary layer hight
BL_p = ps*np.exp(-0.00012*BL_meter)/100
#Hybrid to pressure levels
p = np.zeros(shape = np.shape(tmp_ml))
for k in range(0,len(hybrid)): # Outside for loop? p = [ac/100 + bc * psc for ac, bc, psc in zip(ap,b, ps[:,0,:,:])]
    p[:,k,:,:] = ap[k]/100. + b[k] * ps[:,0,:,:]/100.

#dataset.close()

def Svalbard():
    """
    Set variables needed for focusing on Svalbard
    """
    #########################################
    #SETTING DOMAIN AREA OF MAP
    #########################################
    #latlon=[ 10, 13, 77, 80] # Entire svalbard
    latlon_mapdomain = [11, 14, 78.8, 79.2] #Kingsbay area, domain of interest

    #COORDINATES OF MEASUREMENT SITE
    latlon_old_pier = [11.91929, 78.93030] #end of old pier

    #########################################
    # SETTING DOMAIN AREA OF WANTED GRIDPOINTS
    #########################################
    #DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    lonmin = latlon_mapdomain[0] + 0.7
    lonmax = latlon_mapdomain[1] - 1.1
    latmin = latlon_mapdomain[2] + 0.08
    latmax = latlon_mapdomain[3] - 0.2

    #FIND INDEX AND COORDINATES OF GRIDPOINTS INSIDE THE DOMAIN.
    indx = np.where( [(lon > lonmin) & (lon < lonmax) & (lat >= latmin) & (lat <= latmax)] )
    #Index of points in domain
    jindx_gridPointDomain = indx[1] #index of y/lat
    iindx_gridPointDomain = indx[2] #index of x/lon
    #Coordinate of points in domain
    lons_gridPointDomain = lon[ jindx_gridPointDomain, iindx_gridPointDomain ]
    lats_gridPointDomain = lat[ jindx_gridPointDomain, iindx_gridPointDomain ]

    return latlon_old_pier, latlon_mapdomain, lons_gridPointDomain, lats_gridPointDomain, jindx_gridPointDomain, iindx_gridPointDomain

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
    #Example of conversion if needed later
    ###myProj = Proj("+proj=utm +zone=33 +datum=WGS84")
    ###UTMx, UTMy = myProj(old_pier[0], old_pier[1])
    #######################################
    #PLOT COORDINATES FOR OLDPIER
    plt.plot(latlon_old_pier[0], latlon_old_pier[1], marker='o', markersize=5.0, markeredgewidth=2.5,
                     markerfacecolor='blue', markeredgecolor='blue', zorder=6, transform=crs_latlon)

    #PLOT GRIDPOINT OF INTEREST
    points = plt.plot(lons_gridPointDomain,lats_gridPointDomain, marker='.', markersize=5.0, markeredgewidth=4,
                     markerfacecolor='black', markeredgecolor='black', zorder=6, transform=crs_latlon, linestyle = 'None')

    return projection, crs_latlon

def grid_point(lonp,latp, crs_latlon):
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

def meteogram_vertical(jindx, iindx):
    print("inside meteogram_vertical")

    fig2, ax2 = plt.subplots( figsize = ( 12, 4 ) )
    levels = range( len( hybrid ) )
    lx, tx = np.meshgrid( levels, time_normal[:] )

    #suggestion2
    #CS = plt.pcolormesh(tx, p[:, :, jindx, iindx], specific_humidity_ml[:,:,jindx, iindx])
    # CS = plt.contour(tx, p[:, :, jindx, iindx], specific_humidity_ml[:,:,jindx, iindx]
    #suggestion1
    #CS = plt.pcolormesh(tx, p[:, :, jindx, iindx], tmp_ml[:,:,jindx, iindx])
    cmapback = cm.get_cmap( 'bwr' )
    #tmp_celcius =
    CF = plt.contourf( tx, p[:, :, jindx, iindx], tmp_ml[:, :, jindx, iindx]-273.15, cmap = cmapback )
    plt.clabel( CF, inline = False )
    C = plt.contour( tx, p[:, :, jindx, iindx], specific_humidity_ml[:, :, jindx, iindx], colors = "white", linewidths = 1 )
    plt.clabel( C )

    plt.plot( time_normal[:], BL_p[:, 0, jindx, iindx], color = "black", linewidth=3 ) #(67, 1, 949, 739)
    n= 7
    C = np.sqrt(x_wind_ml[:,::n,jindx, iindx]** 2 + y_wind_ml[:,::n,jindx, iindx]** 2)*1.943844
    cmap = plt.cm.jet
    bounds_ms = [3., 6., 9., 12.]
    bounds_knots = [x*1.943844 for x in bounds_ms]
    norm = mpl.colors.BoundaryNorm(bounds_knots, cmap.N)
    img = plt.barbs(tx[:, ::n], p[:, ::n, jindx, iindx], x_wind_ml[:,::n,jindx, iindx]*1.943844, y_wind_ml[:,::n,jindx, iindx]*1.943844,C,cmap=cmap,norm=norm, length=4)
    cbar = plt.colorbar(img, cmap=cmap, norm = norm, boundaries=bounds_knots, ticks=bounds_knots)
    cbar.ax.set_yticklabels(bounds_ms)  # vertically oriented colorbar

    plt.gca().invert_yaxis()
    xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    ax2.xaxis.set_major_formatter(xfmt)  # Setting xaxis to this format
    #plt.clabel(CS, inline=1, fontsize=10)

    print("Should show plot soon")
    plt.show()
    print("Plot done showed")

def meteogram(jindx, iindx):
    print("e")
    #fig1:temp, dewppint, precip. y1 aksis = temp, y2 aksis = mm precip.
    figm1, axm1 = plt.subplots( figsize = ( 12, 4 ) )
    #figm1.suptitle('test title', fontsize=20)
    axm1_2 = axm1.twinx()
    axm1.plot( time_normal, air_temperature_2m[ :, 0, jindx,iindx] - 273.15 )
    axm1.set_ylabel('2m Temp ($^\circ$C)')
    axm1_2.plot(time_normal, relative_humidity_2m[:, 0, jindx, iindx]*100, color = "green")
    axm1_2.set_ylabel(' 2m Rel. Hum. (%)')

    ##fig2: Wind speed: wind_gust, wind_direction.
    #TODO: When u have height in meters calc: Find Wind at say 80 m height/ height of mountain in areas.
    figm2, axm2 = plt.subplots(figsize=(12, 4))
    #figm2.suptitle('test title', fontsize=20)
    #axm2_2 = axm1.twinx()
    wspeed_gust = np.sqrt( x_wind_gust_10m[:, 0, jindx, iindx] **2 + y_wind_gust_10m[:, 0, jindx, iindx] **2)
    wspeed = np.sqrt( x_wind_10m[:, 0, jindx, iindx] ** 2 + y_wind_10m[:, 0, jindx, iindx] ** 2 )
    axm2.plot(time_normal,wspeed_gust,zorder=0)
    axm2.plot(time_normal, wspeed, zorder=1)
    Q = axm2.quiver(time_normal, wspeed, x_wind_10m[:, 0, jindx, iindx]/wspeed,y_wind_10m[:, 0, jindx, iindx]/wspeed, scale = 80, zorder=2)
    Q_gust = axm2.quiver(time_normal, wspeed_gust, x_wind_gust_10m[:, 0, jindx, iindx]/wspeed_gust, y_wind_gust_10m[:, 0, jindx, iindx]/wspeed_gust, scale = 80, zorder=2)
    axm2.set_ylabel('10m Wind/Gust (m/s)')

    #fig3: precip, preciptype
    #figm3, axm3 = plt.subplots(figsize=(12, 4))

    #fig4: Fluxes radiation : latent, sensible, SW down, LW down, LW_up
    figm4, axm4 = plt.subplots(figsize=(12, 4))
    axm4.plot(time_normal, SH[:, 0, jindx, iindx], zorder=0)
    axm4.plot(time_normal, LH[:, 0, jindx, iindx], zorder=0)
    axm4.set_ylabel('Fluxes (Ws/m$^2$)')

    #fig5: Cloud cover. (maybe merge with fig3 later.)
    plt.show()

def meteogram_average(jindx_gridPointDomain, iindx_gridPointDomain):
    Domain_land = jindx_gridPointDomain[np.where([land_area_fraction])]




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
main()

#plt.show()
