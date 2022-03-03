import matplotlib.pyplot as plt
from loclib.calculation import *
from loclib.check_data import *
from loclib.domain import *  # require netcdf4
from loclib.get_data import *
from mpl_toolkits.basemap import Basemap

##########################################################

modelruntime = "2016120700"
model = "MEPS"
param = ["air_temperature_2m"]
lt = 1
all_available_files = check_available(date=modelruntime, model=model)

file = check_available(date=modelruntime, model=model, param=param)

data_domain = DOMAIN(modelruntime, model, file=file)
data_domain.South_Norway()
lonlat = np.array(data_domain.lonlat)

dmap_meps = DATA(
    model="MEPS",
    data_domain=data_domain,
    param=param,
    file=file,
    step=[0, lt],
    modelrun=modelruntime,
)

dmap_meps.retrieve()

# print(np.shape(dmap_meps.air_temperature_2m))
# PLOT
fig1, ax1 = plt.subplots(figsize=(7, 9))
map = Basemap(
    llcrnrlon=lonlat[0],
    llcrnrlat=lonlat[2],
    urcrnrlon=lonlat[1],
    urcrnrlat=lonlat[3],
    resolution="i",
    projection="lcc",
    lon_0=15.0,
    lat_0=63.3,
    lat_1=63.3,
    area_thresh=0.0001,
)
x, y = map(dmap_meps.longitude, dmap_meps.latitude)
CFW = plt.contourf(
    x,
    y,
    dmap_meps.air_temperature_2m[0, 0, 0, :, :] - 273.15,
    zorder=10,
    alpha=0.9,
    vmin=-30,
    vmax=0,
)
map.drawcoastlines(linewidth=2.0, color="gray", ax=ax1, zorder=1000)
##########################################################
plt.show()
# modelruntime="2016120200"
# model="MEPS"
# data_domain = DOMAIN(modelruntime, model)
# data_domain.South_Norway()
# lonlat = np.array(data_domain.lonlat)
# lt=1
# param = ["air_temperature_2m"]
#
# dmap_meps = DATA(model="MEPS", data_domain=data_domain, param = param, step=[0, lt], modelrun=modelruntime)
# dmap_meps.retrieve()

# PLOT
# fig3, ax3 = plt.subplots(figsize=(7, 9))
# map = Basemap(llcrnrlon=lonlat[0], llcrnrlat=lonlat[2], urcrnrlon=lonlat[1], urcrnrlat=lonlat[3],
#                  resolution='i', projection="lcc",  lon_0=15.0, lat_0=63.3,lat_1 =63.3 , area_thresh=0.0001)
# x, y = map(dmap_meps.longitude, dmap_meps.latitude)
# CFW = plt.contourf(x, y, dmap_meps.air_temperature_2m[0, 0,0, :, :] - 273.15, zorder=10, alpha=0.9,
#        vmin = -30, vmax = 0)
# map.drawcoastlines(linewidth=2.0, color='gray', ax=ax3, zorder=1000)
##########################################################
#
# modelruntime="2020030800"
# model="AromeArctic"
# data_domain = DOMAIN(modelruntime, model)
# data_domain.Svalbard()
# lonlat = np.array(data_domain.lonlat)


# lt=1
# param = ["air_temperature_2m"]
# dmap_aa = DATA(model="AromeArctic", data_domain=data_domain, param = param, step=[0, lt], modelrun=modelruntime)
# dmap_aa.retrieve()

# PLOT

# fig2, ax2 = plt.subplots(figsize=(7, 9))
# map = Basemap(llcrnrlon=lonlat[0], llcrnrlat=lonlat[2], urcrnrlon=lonlat[1], urcrnrlat=lonlat[3],
#                  resolution='i', projection="lcc",  lon_0=-25.0, lat_0=77.5,lat_1 =77.5 , area_thresh=0.0001)
# x, y = map(dmap_aa.longitude, dmap_aa.latitude)
# CFW = plt.contourf(x, y, dmap_aa.air_temperature_2m[0, 0, :, :] - 273.15, zorder=10, alpha=0.9,
#        vmin = -30, vmax = 0)
# map.drawcoastlines(linewidth=2.0, color='gray', ax=ax2, zorder=1000)
#
# plt.show()
