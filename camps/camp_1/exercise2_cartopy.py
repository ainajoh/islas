#exercise2.m
#--------------------------------
# Description
# read and plot netcdf data from AROME
#--------------------------------
# AJ    August 2019
#--------------------------------
##################################################################################################
#Import necessary libraries
##################################################################################################
from netCDF4 import Dataset                     #For reading netcdf files.
import cartopy.crs as ccrs                      #For setting up a map
from cartopy.io import shapereader              #For reading shapefiles containg high-resolution coastline.
from cartopy.feature import NaturalEarthFeature #If low resolution is okey this is a quicker way to draw coastlines
import matplotlib.pyplot as plt                 #For basic plotting in python
import matplotlib.cm as cm                      #For colors on map 
import numpy as np                              #general-purpose array-processing package

#For more information of packages type feks. help(Dataset)
##################################################################################################

##################################################################################################
#READING AROME DATA
##################################################################################################

filename = 'meps_pp_matlab.nc'
#open the netcdf file
ncid = Dataset(filename, "r")
#get info about the file and what variables is in it.
print( ncid )
# get info about one of the variables
print(ncid.variables["air_pressure_at_sea_level"])
#read in variables
prec = ncid.variables["precipitation_amount"][0][:] #time = 0 #Only one time
lon =  ncid.variables["longitude"][:]
lat =  ncid.variables["latitude"][:]
rh = ncid.variables["relative_humidity_2m"][0][0][:] #time = 0, height=0 #only one time and one height
slp = ncid.variables["air_pressure_at_sea_level"][0][0][:] #time=0, height_above_msl=0
t2m = ncid.variables["air_temperature_2m"][0][0][:] #time = 0, height=0 #only one time and one height
#close netcdf file
ncid.close()
###################################################################################################

###################################################################################################
#UNIT SETUP AND DATA PLOTTING
###################################################################################################
# CONVERT UNITS
rh = rh * 100 # rh in %
rh_range = np.arange( 0, 105, 5 )
slp = slp / 100. # Pa to hPa
slp_range = np.arange( 0, 1012, 2 )
t2m = t2m - 273.15# Kelvin to Celcius
# LOAD THE COASTLINE FROM SHAPEFILE
shp = shapereader.Reader('shapefiles/norway/Norway_h_utm.shp')
# DEFINE MAP AREA
#extent = [lon0, lon1, lat0, lat1]
#latlon = [ 4., 6., 60., 61. ] #Byfjorden
latlon=[ -1, 20, 57, 70] #Scandinavia

# SETUP PROJECTION AND MAP ELEMENTS
projection = ccrs.UTM( 32 )
fig = plt.subplots(figsize=(10, 9))
ax = plt.axes(projection=ccrs.UTM( 32 ))
ax.set_extent( latlon )
cmap = cm.get_cmap( 'hot_r' )

#ADD COASTLINE
ax.add_geometries( shp.geometries(), ccrs.UTM(32), facecolor='none', edgecolor='black', zorder=2 )
#SET GRIDLINES
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.3, linestyle='--')
#plot the RH variable, semi-transparent
#CS = ax.pcolormesh( lon, lat, rh, transform=ccrs.PlateCarree(),vmin=7,vmax=17, cmap="hot_r" )
CS = ax.pcolormesh( lon, lat, t2m, transform=ccrs.PlateCarree(), vmin=7,vmax=17, cmap=cmap,zorder=1, alpha=0.6, linewidth=0.0015625, antialiased=True)
#plot pressure contours on top
plt.contour(lon,lat,slp,slp_range,transform=ccrs.PlateCarree(), zorder=3, colors="k", linewidths=0.6)

#Make colorbars adjusted to CS
cbar = plt.colorbar(CS)
#Add labels
cbar.set_label("Temperature (C)")
plt.savefig("AROME_PLOT.png")
plt.show()


