from loclib.domain import *  # require netcdf4
from loclib.check_data import *
from loclib.get_data import *
from loclib.calculation import *
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


##########################################################
YYYY = ["2020"]
MM = ["03"]
DD = [
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
]
HH = ["12"]  # , "06", "12", "18"]
model = "MEPS"
# param = ["air_temperature_2m"]
# param = ["air_temperature_ml"]
param = ["air_temperature_2m", "air_temperature_ml"]
levtype = "ml"
mbrs = None
lt = 1

for yy in YYYY:
    for mm in MM:
        for dd in DD:
            for hh in HH:
                print("loop start")
                modelruntime = f"{yy}{mm}{dd}{hh}"
                print(modelruntime)
                file = check_available(
                    date=modelruntime,
                    model=model,
                    levtype=levtype,
                    param=param,
                    mbrs=mbrs,
                )
                print(file)
                data_domain = DOMAIN(modelruntime, model, file=file)
                data_domain.South_Norway()
                lonlat = np.array(data_domain.lonlat)
                dmap_meps = DATA(
                    model="MEPS",
                    data_domain=data_domain,
                    file=file,
                    levtype=levtype,
                    param=param,
                    step=[0, lt],
                    modelrun=modelruntime,
                    mbrs=mbrs,
                )
                dmap_meps.retrieve()

                print(np.shape(dmap_meps.__getattribute__(param[0])))
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
                    dmap_meps.air_temperature_ml[0, 0, 0, :, :] - 273.15,
                    zorder=10,
                    alpha=0.9,
                    vmin=-30,
                    vmax=0,
                )
                # map.drawcoastlines(linewidth=2.0, color='gray', ax=ax1, zorder=1000)
                print("loop done")
##########################################################
# plt.show()
