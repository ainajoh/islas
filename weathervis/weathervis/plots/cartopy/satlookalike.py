
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from weathervis.config import *
from weathervis.utils import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.check_data import *
from weathervis.calculation import *
#param = ["toa_outgoing_longwave_flux"]

from netCDF4 import Dataset
ss = Dataset("https://thredds.met.no/thredds/dodsC/meps25epsarchive/2018/03/17/meps_subset_2_5km_20180317T00Z.nc")
param = ["toa_outgoing_longwave_flux"]
model = "MEPS"
date = "2020100212"
step=[0]
s = check_data(model = model,param = param, date=date, step=step)
#s = check_data(model = model, date=date)

myfile = s.file

data = get_data(model = model, param=param, date=date, file = myfile, step=step)
data.retrieve()

lon0 = data.longitude_of_central_meridian_projection_lambert
lat0 = data.latitude_of_projection_origin_projection_lambert
parallels = data.standard_parallel_projection_lambert
# setting up projection
globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                               globe=globe)
def impact_OLR(data):
    ax = plt.subplot(projection=crs)
    ax.coastlines('10m')
    ax.pcolormesh(data.x,data.y,data.toa_outgoing_longwave_flux[0,0,:,:],vmin=-230,vmax=-110,cmap=plt.cm.Greys_r)
    make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
    plt.savefig(make_modelrun_folder + "/{0}_OLR_{1}_{2:02d}.png".format(model, dt, tim), bbox_inches="tight",
                 dpi=200)


impact_OLR(data)