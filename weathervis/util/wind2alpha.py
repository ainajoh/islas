from netCDF4 import Dataset as NetCDFFile
import math

# ******************************************************************************
# This routine calculates the local grid rotation (alpha) from input file,
# and writes to a separate output file.
# Formula:
#   alpha = atan2(dlatykm,dlonykm)*180/pi - 90)
#
# Wind direction relative to Earth (wdir) may later be calculated as follows:
#   wdir = alpha + 90-atan2(v,u)
# where u and v are model wind relative to model grid
#
# ******************************************************************************

# In and outfiles
# url = "http://thredds.met.no/thredds/dodsC/meps25files/meps_det_extracted_2_5km_latest.nc"
#infile = "http://thredds.met.no/thredds/dodsC/meps25epsarchive/2016/11/29/meps_subset_2_5km_20161129T00Z.nc"
#infile = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_full_2_5km_latest.nc"
infile = "https://thredds.met.no/thredds/dodsC/aromearcticarchive/2020/02/20/arome_arctic_extracted_2_5km_20200220T00Z.nc?"
outfile = "alpha_full_AA.nc"
#infile = "https://thredds.met.no/thredds/dodsC/mepslatest/meps_lagged_6_h_latest_2_5km_latest.nc"
#infile = "https://thredds.met.no/thredds/dodsC/meps25epsarchive/2020/02/20/meps_det_2_5km_20200220T00Z.nc"
#outfile = "alpha_full_MEPS.nc"



def distance(origin, destination):
    """
    (Source: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude)

    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# *****************************************************************************

nc = NetCDFFile(infile)

# Variables
xx = nc.variables['x'][:]
yy = nc.variables['y'][:]
lat = nc.variables['latitude'][:]
lon = nc.variables['longitude'][:]
alpha = lat  # Target matrix

for j in range(0, yy.size - 1):
    for i in range(0, xx.size - 1):
        # Prevent out of bounds
        if j == yy.size - 1:
            j1 = j - 1;
            j2 = j
        else:
            j1 = j;
            j2 = j + 1
        if i == xx.size - 1:
            i1 = i - 1;
            i2 = i
        else:
            i1 = i;
            i2 = i + 1

        dlatykm = distance([lat[j1, i1], lon[j1, i1]], [lat[j2, i1], lon[j1, i1]])
        dlonykm = distance([lat[j1, i1], lon[j1, i1]], [lat[j1, i1], lon[j2, i1]])
        print(dlatykm)
        print(dlonykm)
        alpha[j, i] = math.atan2(dlatykm, dlonykm) * 180 / math.pi - 90

# Make NetCDF file
rg = NetCDFFile(outfile, "w", format="NETCDF4")
x = rg.createDimension("x", xx.size)
y = rg.createDimension("y", yy.size)
alph = rg.createVariable("alpha", "f4", ("y", "x"))
alph[:] = alpha
rg.close()