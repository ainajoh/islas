import numpy as np
from netCDF4 import Dataset                     #For reading netcdf files.

def data_domain(latlon_mapdomain=[11, 14, 78.8, 79.2], adjusted_domain = [0.7, -1.1, 0.08, -0.2]):
    #Smaller file with less data, but only want latlon. if problems maybe use "full" not sfx
    url ="https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
    dataset = Dataset(url)

    lon = dataset.variables["longitude"][:]
    lat = dataset.variables["latitude"][:]
    # DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    latlon_datadomain = [a + b for a, b in zip(latlon_mapdomain, adjusted_domain)]

    idx = np.where((lon > latlon_datadomain[0]) & (lon < latlon_datadomain[1]) & \
                   (lat >= latlon_datadomain[2]) & (lat <= latlon_datadomain[3]))

    #jindx_gridPointDomain = indx[0]  # index of y/lat
    #iindx_gridPointDomain = indx[1]  # index of x/lon
    dataset.close()
    return idx, latlon_datadomain

