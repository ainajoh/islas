import cartopy.crs as ccrs
import cartopy.feature as cfeature
from lagranto.plotting import plot_trajs
import matplotlib.pyplot as plt
from lagranto import Tra

path = "/Users/ainajoh/Data/ISLAS/flexpart/test/"
times = ["20190320_00"]
for t in times:
    filein = path + "lsl" + t
print(filein)
trajs = Tra()
trajs.load_ascii(filein)

print(trajs)
wcb_trajs = trajs

#wcb_trajs = Tra()
#wcb_trajs.set_array(trajs[wcb_index[0], :])


crs = ccrs.Stereographic(central_longitude=10, central_latitude=90, true_scale_latitude=90)

fig = plt.figure()
ax = plt.axes(projection=crs)
#ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-90, 40, 50, 90])  # [lon0,lon1, lat0, lat1]
cax = plot_trajs(ax, wcb_trajs, "Q")  # color with Q
    # cax = plot_trajs( ax, upt_idx, "Q" ) #color with Q
    # an = plt.annotate(trajs["time"])
    # for i in range(len(x)):
    #    plt.annotate(labls[i], xy=(x[i,2], y[i,2]), rotation=rotn[i,2])
cbar = fig.colorbar(cax)

plt.show()


