from loclib.config import *  # require netcdf4
from loclib.domain import *  # require netcdf4
from loclib.get_data import *
from loclib.calculation import *
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


lt = 24  # lead time.
modelruntime = "latest"

param_ML = ["air_temperature_ml"]#, "specific_humidity_ml", "x_wind_ml", "y_wind_ml"]
param_SFC = ["air_temperature_2m"]#, "surface_air_pressure", "air_pressure_at_sea_level",
#             "surface_geopotential", "atmosphere_boundary_layer_thickness", "relative_humidity_2m",
#             "x_wind_gust_10m", "y_wind_gust_10m", "x_wind_10m", "y_wind_10m", "specific_humidity_2m",
#             "precipitation_amount_acc", "land_area_fraction"]
param_sfx = ["SST", "H", "LE"]

print("\n###########################\n"
      "\n# SET DOMAIN               #\n"
      "\n###########################\n")
data_domain = DOMAIN()
data_domain.Finse()
sites = pd.read_csv("bin/sites.csv", sep=";", header=0, index_col=0)
ZeppelinObservatory = sites.loc["ZeppelinObservatory"]
OldPier = sites.loc["OldPier"]

print("\n###########################\n"
      "\n# RETRIEVE DATA            #\n"
      "\n###########################\n")
########################################
# RETRIEVE
dmet = DATA(model="MEPS", data_domain=data_domain, param_SFC=param_SFC, param_ML=param_ML, fctime=[0, lt], modelrun=modelruntime)
dmet.retrieve()
dmet_sfx = DATA(data_domain=data_domain, param_sfx=param_sfx, fctime=[0, lt], type="sfx", modelrun=modelruntime)
dmet_sfx.retrieve()

print("\n###########################\n"
      "\n# CALCULATE and set DATA          #\n"
      "\n###########################\n")

time_normal = timestamp2utc(dmet.time)
modelrun = timestamp2utc([dmet.forecast_reference_time])

precip1h = precip_acc(dmet.precipitation_amount_acc, acc=1)
precip3h = precip_acc(dmet.precipitation_amount_acc, acc=3)
#future speedup.. maybe do it for only points needed? But units changes as it is used for display later.
p = ml2pl(dmet.ap, dmet.b, dmet.surface_air_pressure)
theta = potential_temperatur(dmet.air_temperature_ml, p)
specific_humidity_ml = dmet.specific_humidity_ml * 1000. #g/kg
geotoreturn = ml2alt_sl(p, dmet.surface_geopotential, dmet.air_temperature_ml, dmet.specific_humidity_ml)
heighttoreturn = ml2alt_gl(p, dmet.air_temperature_ml, dmet.specific_humidity_ml)
t_v_level = virtual_temp(dmet.air_temperature_ml, dmet.specific_humidity_ml)
dtdz = lapserate(dmet.air_temperature_ml, heighttoreturn)
density_ml = density( t_v_level, dmet.surface_air_pressure)
sample_ml = get_samplesize(specific_humidity_ml, density_ml, acc=3)

#density_2m = density( t_v_level, dmet.surface_air_pressure)
#sample_2m = get_samplesize(specific_humidity_ml, density_ml, acc=3)

def point_calculation(jindx, iindx):
    height = np.full(np.shape(dmet.specific_humidity_ml[:, :, jindx, iindx]), float(ZeppelinObservatory.height))
    ZeppelinObservatory_height_pl = \
        point_alt_sl2pres(jindx, iindx,
                          height,
                          geotoreturn, t_v_level, p, dmet.surface_air_pressure,
                          dmet.surface_geopotential)

    h_gl = dmet.atmosphere_boundary_layer_thickness[:, 0, jindx, iindx]
    h_sl = h_gl + dmet.surface_geopotential[:, 0, jindx, iindx] / 9.08
    #

    h = np.repeat(h_sl, repeats=len(dmet.hybrid), axis=0).reshape(np.shape(specific_humidity_ml[:, :, jindx, iindx]))
    #h = np.repeat(h_sl, repeats=len(dmet.hybrid), axis=0).reshape(np.shape(specific_humidity_ml[:, :, jindx, iindx]))

    BL = point_alt_sl2pres(jindx, iindx,
                           h,
                           geotoreturn, t_v_level, p, dmet.surface_air_pressure,
                           dmet.surface_geopotential)


    return ZeppelinObservatory_height_pl, BL

def plot_meteogram_vertical(jindx, iindx, dirName_b1, figname_b1, ip):
    ZeppelinObservatory_height_pl, BL = point_calculation(jindx, iindx)

    # Correct unit for proper display
    p[:, :, jindx, iindx] = p[:, :, jindx, iindx] / 100
    ZeppelinObservatory_height_pl = ZeppelinObservatory_height_pl / 100.
    BL = BL / 100.
    #################################

    print("\n###########################\n"
          "\nINITIALISING PLOTTING: meteogram_vertical \n"
          "\n###########################\n")

    figm1, (axm1, axm2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), sharex=True)
    plt.subplots_adjust(wspace=0.001)

    levels = range(len(dmet.hybrid))
    lx, tx = np.meshgrid(levels, time_normal[:])


    #################################
    # P1: spec.hum with lapserate and BLheight
    #################################
    # Ground color gray
    paths = axm1.fill_between(time_normal[:], dmet.surface_air_pressure[:, 0, jindx, iindx] / 100, \
                              dmet.air_pressure_at_sea_level[:, 0, jindx, iindx] / 100, color="gray")
    #spec humidity
    cmap = cm.get_cmap('gnuplot2_r')  # BrBu  BrYlBu
    lvl = np.linspace(np.min(specific_humidity_ml[:, :, jindx, iindx]),
                      np.max(specific_humidity_ml[:, :, jindx, iindx]), 20)
    CF_Q = axm1.contourf(tx, p[:, :, jindx, iindx], specific_humidity_ml[:, :, jindx, iindx], levels=lvl, cmap=cmap)
    axins1 = inset_axes(axm1, width='80%', height='15%',
                        bbox_to_anchor=(0.87, 0.84, 0.13, 0.15),
                        bbox_transform=axm1.transAxes,
                        loc="upper center")
    #Zeppelin height
    Z = axm1.plot(time_normal[:], ZeppelinObservatory_height_pl, linewidth=3, color="c", zorder=1000, alpha=0.7)
    axm1.annotate("Zeppelin = 479m", (time_normal[1], ZeppelinObservatory_height_pl[1]), color="c", zorder=1000,
                  xytext=(0, 2), textcoords="offset points")
    #BL height
    h_gl = dmet.atmosphere_boundary_layer_thickness[:, 0, jindx, iindx]
    h_sl = h_gl + dmet.surface_geopotential[:, 0, jindx, iindx] / 9.08
    P_BL = axm1.plot(time_normal[:], BL, "X-", color="black", linewidth=3, zorder=1000)  # (67, 1, 949, 739)
    axm1.annotate("BLH", (time_normal[1], BL[1]), color="black", zorder=1000)
    for i in range(0,np.shape(h_sl)[0], 4 ):
        axm1.annotate('{0:.0f}m'.format(h_sl[i]), (time_normal[i], BL[i]), color="black",
                      zorder=1000, xytext=(0, 5), textcoords="offset points", ha='center')
    #Lapserate
    lvl1 = np.linspace(-15, -9.8, 5)
    lvl2 = np.linspace(-6.5, -0.5, 5)
    lvl3 = np.linspace(0, 10, 5)
    lvl = np.append(lvl1, lvl2)
    lvl = np.append(lvl, lvl3)
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    CL = axm1.contour(tx, p[:, :, jindx, iindx], dtdz[:, :, jindx, iindx], levels=lvl, colors="gray", linewidth=1,
                     zorder=100, )
    unstable = CL.collections[0:5]
    plt.setp(unstable, color="r", zorder=100)
    stableinv = CL.collections[10:15]
    plt.setp(stableinv, color="lime", zorder=100)

    axm1.clabel(CL, inline=False, fmt='%.1f' + r"K/km")

    #label
    custom_lines = [Line2D([0], [0], color="r", lw=2),
                    Line2D([0], [0], color="lime", lw=2),
                    Line2D([0], [0], color="gray", lw=2)]

    axm1.legend([Z[0], P_BL[0], custom_lines[0], custom_lines[1], custom_lines[2]],
                ['Zeppelin = 479m', 'BLH', 'Unstable', 'Very Stable', 'Stable'], loc='upper left').set_zorder(99999)
    axm1.invert_yaxis()
    axm1.set_ylabel("Pressure [hPa]")
    axm1.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 850)
    #################################
    # P2: Potential temp with wind
    #################################
    #Ground in gray
    paths = axm2.fill_between(time_normal[:], dmet.surface_air_pressure[:, 0, jindx, iindx] / 100, \
                              dmet.air_pressure_at_sea_level[:, 0, jindx, iindx] / 100, color="gray")
    #Wind
    n = 4
    wspeed = np.sqrt(dmet.x_wind_ml[:, :, jindx, iindx] ** 2 + dmet.y_wind_ml[:, :, jindx, iindx] ** 2)

    cmap = plt.cm.RdYlBu_r#plt.cm.jet RdYlBu
    axm2.barbs(tx[::n, ::n], p[::n, ::n, jindx, iindx], dmet.x_wind_ml[::n, ::n, jindx, iindx] * 1.943844,
                     dmet.y_wind_ml[::n, ::n, jindx, iindx] * 1.943844, length=5, zorder=1000)
    lvl = np.arange(0,40,3)
    CF_WS = axm2.contourf( tx[:, :], p[:, :, jindx, iindx], wspeed, cmap = cmap, alpha=0.8, zorder=10, levels=lvl, vmin = 0, vmax= 40)
    axins2 = inset_axes(axm2, width='80%', height='15%',
                        bbox_to_anchor=(0.87, 0.84, 0.13, 0.15),
                        bbox_transform=axm2.transAxes,
                        loc="upper center")

    #potential temp
    lvl = np.linspace(np.min(theta[:, :, jindx, iindx]), np.max(theta[:, :, jindx, iindx]), 90)
    CS = axm2.contour(tx, p[:, :, jindx, iindx], theta[:, :, jindx, iindx], colors="black", levels=lvl, zorder=900)
    axm2.clabel(CS, [*CS.levels[2:5:1], *CS.levels[5:10:2], *CS.levels[15:20:5]], inline=True,
                fmt='$\Theta$ = %1.0fK')  # '%1.0fK')
    #zeppelin height
    Z = axm2.plot(time_normal[:], ZeppelinObservatory_height_pl, linewidth=3, color="c", zorder=1000, alpha=0.7)
    axm2.annotate("Zeppelin = 479m", (time_normal[1], ZeppelinObservatory_height_pl[1]), color="c", zorder=1000,
                  xytext=(0, 2), textcoords="offset points")

    #label
    axm2.legend([Z[0], CS],['Zeppelin = 479m',"Pot. Temp."], loc='upper left').set_zorder(99999)
    axm2.invert_yaxis()
    axm2.set_ylabel("Pressure [hPa]")
    axm2.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 850)

    #################################
    # ADJUSTMENTS AND LABELS
    #################################
    xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    axm1.xaxis.set_major_formatter(xfmt)  # Setting xaxis to this format
    axm2.xaxis.set_major_formatter(xfmt)  # Setting xaxis to this format

    axm1.add_patch(plt.Rectangle((0.87, 0.84), 0.13, 0.15, fc=[1, 1, 1, 0.7],
                                transform=axm1.transAxes, zorder=1000))
    ticks = np.linspace(0, 1.4, 4)
    cbar = plt.colorbar(CF_Q, extend="both", cax=axins1, orientation="horizontal", ticks=ticks, format='%.1f')
    cbar.ax.xaxis.set_tick_params(pad=-0.5)
    cbar.set_label('Spec. Hum. [g/kg]', labelpad=-0.5)
    plt.rcParams["axes.axisbelow"] = True

    axm2.add_patch(plt.Rectangle((0.87, 0.84), 0.13, 0.15, fc=[1, 1, 1, 0.7],
                                 transform=axm2.transAxes, zorder=1000))
    ticks = np.linspace(0, 40, 4)
    cbar = plt.colorbar(CF_WS, extend="both", cax=axins2, orientation="horizontal", ticks=ticks, format='%.0f')
    cbar.ax.xaxis.set_tick_params(pad=-0.5)
    cbar.set_label('Wind Speed [m/s]', labelpad=-0.5)
    plt.rcParams["axes.axisbelow"] = True

    #################################
    # SET ADJUSTMENTS ON AXIS
    #################################
    xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_maj = mdates.DateFormatter('%d.%m')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_min = mdates.DateFormatter('%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

    axm2.xaxis.set_major_locator(mdates.DayLocator())
    axm2.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
    axm2.xaxis.set_major_formatter(xfmt_maj)
    axm2.xaxis.set_minor_formatter(xfmt_min)

    axm1.xaxis.grid(True, which="major", linewidth=2)
    axm1.xaxis.grid(True, which="minor", linestyle="--")
    axm2.xaxis.grid(True, which="major", linewidth=2)
    axm2.xaxis.grid(True, which="minor", linestyle="--")
    axm2.tick_params(axis="x", which="major", pad=12)

    figm1.tight_layout()
    plt.savefig(dirName_b1 + figname_b1 + "_LOC" + str(ip) +
                "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[jindx, iindx],
                                                dmet.latitude[ jindx, iindx]) + ".png")
    plt.close()


def plot_meteogram(jindx, iindx, dirName_b0, figname_b0, ip ):
    #################################

    print("\n###########################\n"
          "\nINITIALISING PLOTTING: meteogram \n"
          "\n###########################\n")
    figm2, (axm1, axm2, axm3, axm4) = plt.subplots(nrows=4, ncols=1, figsize=(12, 14), sharex=True)
    def autolabel(rects, axis, fmt = '{0:.1f}', space=2):  # for the precip. Got from e
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects[::space]:
            height = rect.get_height()
            axis.annotate(fmt.format(height).strip('-').strip('0'),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    #################################
    # P1: TEMP and RH and PRECIP
    #################################
    #temp
    T2M=axm1.plot(time_normal, dmet.air_temperature_2m[:, 0, jindx, iindx] - 273.15, color="red")
    TML0=axm1.plot(time_normal, dmet.air_temperature_ml[:, -1, jindx, iindx] - 273.15, "--", color="red")
    axm1.set_ylabel('Temp ($^\circ$C)', color="red")
    axm1.tick_params(axis="y", colors="red")
    axm1.set_ylim(bottom=-25, top=0)
    #precip
    axm1_3 = axm1.twinx()
    wd = -0.125
    P_bar = axm1_3.bar(time_normal, precip1h[:, 0, jindx, iindx], color="blue", alpha=0.8, width=wd/4, align="edge")
    axm1_3.set_ylim(bottom = 0, top = 10)
    autolabel(P_bar, axm1_3, space=2)
    axm1_3.set_yticks([])
    axm1_3.set_xticks([])
    #RH
    axm1_2 = axm1.twinx()
    axm1_2.plot(time_normal, dmet.relative_humidity_2m[:, 0, jindx, iindx] * 100, color="seagreen", alpha=0.8)
    axm1_2.set_ylabel(' 2m Rel. Hum. (%)', color="seagreen")
    axm1_2.set_ylim(bottom = 0.001, top = 100)
    axm1_2.tick_params(axis="y", colors="seagreen")
    #label
    axm1.legend([T2M[0], TML0[0],P_bar[0]], ["2m Temp.", "ml0 Temp.","1h acc precip"], loc='upper left').set_zorder(99999)

    #################################
    # P2: SENSIBLE HEAT, sample size
    #################################
    #specifichum
    Q2M=axm2.plot(time_normal, dmet.specific_humidity_2m[:, 0, jindx, iindx] * 1000, zorder=0, color="green", alpha=0.8)
    QML0=axm2.plot(time_normal, dmet.specific_humidity_ml[:, -1, jindx, iindx] * 1000,"--", zorder=0, color="green", alpha=0.8)
    axm2.set_ylabel('Spec. Hum. (g/kg)', color="green")
    axm2.tick_params(axis="y", colors="green")
    axm2.set_ylim(bottom=0, top=1.5)
    #precip
    axm2_3 = axm2.twinx()
    wd = -0.125 #width of 3h x axis found emprically, should heve better automatic
    S_bar = axm2_3.bar(time_normal[0::3], sample_ml[0::3, -1, jindx, iindx], ls = "--", ec="green",color="lightblue", alpha=0.8, width=wd, align="edge")
    autolabel(S_bar, axis=axm2_3, fmt='{:.3f}', space=1)
    axm2_3.set_ylim(bottom=0, top=0.3)
    axm2_3.set_ylabel(' Sample size (g)')
    #label
    axm2.legend([Q2M[0], QML0[0],S_bar[0]], ["2m Spec.Hum..", "ml0 Spec.Hum.", "3h acc samplesize"], loc='upper left').set_zorder(99999)

    #################################
    # P3: Wind, pressure.
    #################################
    axm3_2 = axm3.twinx()
    #wind
    wspeed_gust = np.sqrt(dmet.x_wind_gust_10m[:, 0, jindx, iindx] ** 2 + dmet.y_wind_gust_10m[:, 0, jindx, iindx] ** 2)
    wspeed = np.sqrt(dmet.x_wind_10m[:, 0, jindx, iindx] ** 2 + dmet.y_wind_10m[:, 0, jindx, iindx] ** 2)
    GUST = axm3.plot(time_normal, wspeed_gust, zorder=0, color = "magenta")
    WIND=axm3.plot(time_normal, wspeed, zorder=1, color = "darkmagenta")
    axm3.quiver(time_normal, wspeed, dmet.x_wind_10m[:, 0, jindx, iindx] / wspeed,
                    dmet.y_wind_10m[:, 0, jindx, iindx] / wspeed, scale=80, zorder=2)

    axm3.quiver(time_normal, wspeed_gust, dmet.x_wind_gust_10m[:, 0, jindx, iindx] / wspeed_gust,
                        dmet.y_wind_gust_10m[:, 0, jindx, iindx] / wspeed_gust, scale=80, zorder=2)
    axm3.set_ylabel('wind (m/s)')
    axm3.set_ylim(bottom=0, top=25)
    axm3.tick_params(axis="y", color = "darkmagenta")
    #pressure
    axm3_2.plot(time_normal, dmet.surface_air_pressure[:, 0, jindx, iindx] / 100, color="k")
    axm3_2.set_ylabel(' Surface Pressure (hPa)')
    #axm3_2.set_ylim(bottom=900, top=1050)
    #label
    axm3.legend([GUST[0], WIND[0]], ["10m wind gust", "10m wind (10min mean)"], loc='upper left').set_zorder(99999)

    #################################
    # P3: FLUXES
    #################################
    P_SH = axm4.plot(time_normal, dmet_sfx.H[:, jindx, iindx], zorder=0, color="blue")
    P_LH = axm4.plot(time_normal, dmet_sfx.LE[:, jindx, iindx], zorder=0, color="orange")
    axm4.set_ylabel('Fluxes (W/m$^2$)')
    axm4.legend([P_SH[0], P_LH[0]], ["Sensible Heat Flux", "Latent Heat Flux"], loc='upper left').set_zorder(99999)
    #################################
    # SET ADJUSTMENTS ON AXIS
    #################################
    xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_maj = mdates.DateFormatter('%d.%m')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    xfmt_min = mdates.DateFormatter('%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

    axm4.xaxis.set_major_locator(mdates.DayLocator())
    axm4.xaxis.set_minor_locator(mdates.HourLocator((0,6,12,18)))
    axm4.xaxis.set_major_formatter(xfmt_maj)
    axm4.xaxis.set_minor_formatter(xfmt_min)

    axm1.xaxis.grid(True, which = "major", linewidth=2)
    axm1.xaxis.grid(True, which = "minor", linestyle="--" )
    axm2.xaxis.grid(True, which = "major", linewidth=2)
    axm2.xaxis.grid(True, which = "minor", linestyle="--" )
    axm3.xaxis.grid(True, which = "major", linewidth=2)
    axm3.xaxis.grid(True, which = "minor", linestyle="--" )
    axm4.xaxis.grid(True, which = "major", linewidth=2)
    axm4.xaxis.grid(True, which = "minor", linestyle="--" )
    axm4.tick_params(axis="x", which="major", pad=12)

    figm2.tight_layout()
    plt.savefig(dirName_b0 + figname_b0 + "_LOC" + str(ip) +
                "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[jindx, iindx],
                                                dmet.latitude[jindx, iindx]) + ".png")

    plt.close()


    print("\n###########################\n"
          "\nDONE meteogram \n"
          "\n###########################\n")


######## MAP ##############################################
def background_map(lonlat,ax):
    map = Basemap(llcrnrlon=lonlat[0], llcrnrlat=lonlat[2], urcrnrlon=lonlat[1], urcrnrlat=lonlat[3],
                  resolution='f', projection="tmerc", lon_0=15., lat_0=42.,
                  area_thresh=0.0001)  # "epsg=5973,
    map.drawmapboundary(fill_color='lightskyblue')

    map.readshapefile('./bin/shapefiles/svalbard/S100_Land_f_WGS84', 'S100_Land_f_WGS84',
                      zorder=1000, linewidth=2, color = "k")
    patches=[]
    for info, shape in zip(map.S100_Land_f_WGS84_info, map.S100_Land_f_WGS84):
        patches.append(Polygon(np.array(shape), True))

    ax.add_collection(PatchCollection(patches, facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2))
    #map.readshapefile('./bin/shapefiles/svalbard/S100_Land_f_WGS84', 'S100_Land_f_WGS84',
    #                  zorder=1000, linewidth=2)

    x, y = map(OldPier.lon, OldPier.lat)

    xlon,ylon = map(dmet.longitude, dmet.latitude)
    CC = plt.contour(xlon,ylon,dmet.land_area_fraction[0,0,:,:], alpha = 0.6, zorder=3,levels=[0.9, 1, 1.1],
                      colors="b", linewidths=5)
    #ax.clabel(CC, inline=False, fmt=r"Models coastline", colors="b", fontsize="medium")
    #plt.annotate('some text here', (1.4, 1.6))

    #plt.pcolormesh( xlon,ylon,dmet.land_area_fraction[0,0,:,:], cmap = "binary", alpha = 0.5, zorder=3)

    plt.plot(x, y, marker='o', markersize=5.0, markeredgewidth=2.5,
             markerfacecolor='blue', markeredgecolor='blue', zorder=1000)
    x, y = map(dmet.longitude, dmet.latitude)

    points = plt.plot(x, y, marker='.', markersize=5.0, markeredgewidth=4,
                      markerfacecolor='black', markeredgecolor='black', zorder=1000, linestyle='None')
    return map
def plot_maplocation(point, dirName_b2, figname_b2):
    map_domain = DOMAIN()
    map_domain.KingsBay_Z0()
    lonlat = np.array(map_domain.lonlat)
    ip = 0
    for p in point:
        figm2, ax = plt.subplots(figsize=(12, 14))
        map = background_map(lonlat,ax)
        x, y = map(dmet.longitude, dmet.latitude)
        plt.plot(x[p], y[p], marker='.', markersize=6.0, markeredgewidth=4,
                markerfacecolor='red', markeredgecolor='red', zorder=1000, linestyle='None')
        figname_b2_2 = figname_b2 + "_LOC" + str(ip) + \
                       "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[p],dmet.latitude[p])
        figm2.tight_layout()
        plt.savefig(dirName_b2 + figname_b2_2 + ".png")

        plt.plot(x, y, marker='.', markersize=6.0, markeredgewidth=4,
                markerfacecolor='black', markeredgecolor='black', zorder=1000, linestyle='None')
        ip += 1
        plt.close() #Close to not just plot ontop of eachother.
##########################################################

def closest(lat, lon):
    ind_list = []
    dlat = np.abs(dmet.latitude - lat)
    dlon = np.abs(dmet.longitude - lon)
    ddd = np.add(dlat, dlon)
    for i in range(0, np.shape(ddd)[0] * np.shape(ddd)[1]):
        ind = np.unravel_index(np.nanargmin(ddd, axis=None), ddd.shape)
        ddd[ind] = np.nan
        ind_list.append(ind)

    return ind_list

def setup_directory():
    projectpath = "../../output/"
    figname = "fc_" + modelrun[0].strftime('%Y%m%d%H')
    # dirName = projectpath + "result/" + modelrun[0].strftime('%Y/%m/%d/%H/')
    dirName = projectpath + "result/" + "fc_" + modelrun[0].strftime('%Y%m%d/')

    dirName_b1 = dirName + "met/"
    figname_b1 = "vmet_" + figname

    dirName_b0 = dirName + "met/"
    figname_b0 = "met_" + figname

    dirName_b2 = dirName + "map/"
    figname_b2 = "map_" + figname

    dirName_b3 = dirName + "met/"
    figname_b3 = "amet_" + figname

    if not os.path.exists(dirName_b1):
        os.makedirs(dirName_b1)
        print("Directory ", dirName_b1, " Created ")
    else:
        print("Directory ", dirName_b1, " already exists")
    if not os.path.exists(dirName_b2):
        os.makedirs(dirName_b2)
        print("Directory ", dirName_b2, " Created ")
    else:
        print("Directory ", dirName_b2, " already exists")
    return dirName_b0, dirName_b1, dirName_b2,dirName_b3, figname_b0, figname_b1, figname_b2,figname_b3

def plot_amaplocation(point, dirName_b2, figname_b2, sitename):
    map_domain = DOMAIN()
    map_domain.KingsBay_Z0()
    lonlat = np.array(map_domain.lonlat)
    figma2, ax = plt.subplots(figsize=(12, 14))
    map = background_map(lonlat,ax)
    x, y = map(dmet.longitude, dmet.latitude)

    plt.plot(x[point], y[point], marker='.', markersize=6.0, markeredgewidth=4,
             markerfacecolor='red', markeredgecolor='red', zorder=1000, linestyle='None')
    figname_b2_2 = figname_b2 + "_LOC["+sitename+"]"
    figma2.tight_layout()
    plt.savefig(dirName_b2 + figname_b2_2 + ".png")
    plt.close() #Close to not just plot ontop of eachother.
##########################################################
def meteogram_average(indx, dirName_b2 ,figname_b2, sitename):
    #lona = dmet.longitude[indx[0], indx[1]]
    #lata = dmet.latitude[indx[0], indx[1]]
    figma1, (axma1, axma2,axma3, axma4) = plt.subplots(nrows=4, ncols=1, figsize=(12, 14), sharex=True)
    def autolabel(rects, axis, fmt = '{0:.1f}', space=2):  # for the precip. Got from e
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects[::space]:
            height = rect.get_height()
            axis.annotate(fmt.format(height).strip('-').strip('0'),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', zorder=99999)
    #################################
    # P1: temp, rh, precip
    #################################
    #temp
    temp2m_mean = np.mean(dmet.air_temperature_2m[:, 0, indx[0], indx[1]], axis=(1))

    T2M_MEAN = axma1.plot(time_normal, temp2m_mean - 273.15, color="red", linewidth=3)
    T2M = axma1.plot(time_normal, dmet.air_temperature_2m[:, 0, indx[0], indx[1]] - 273.15, color="red", linewidth=0.2, alpha=0.7)

    axma1.set_ylabel('Temp ($^\circ$C)', color="red")
    axma1.tick_params(axis="y", colors="red")
    axma1.set_ylim(bottom=-25, top=0)
    #rh
    axma1_2 = axma1.twinx()
    relhum2m_mean = np.mean(dmet.relative_humidity_2m[:, 0, indx[0], indx[1]], axis=(1))
    RH2m_MEAN =axma1_2.plot(time_normal, relhum2m_mean*100, color="seagreen", linewidth=3)
    RH2M = axma1_2.plot(time_normal, dmet.relative_humidity_2m[:, 0, indx[0], indx[1]]*100, color="seagreen", linewidth=0.2, alpha=0.7)
    axma1_2.set_ylabel(' 2m Rel. Hum. (%)', color="seagreen")
    axma1_2.set_ylim(bottom=0.001, top=100)
    axma1_2.tick_params(axis="y", colors="seagreen")
    #precip
    axma1_3 = axma1.twinx()
    wd = -0.125
    precip_mean = np.mean(precip1h[:, 0, indx[0], indx[1]], axis=(1))
    precip_max = np.max(precip1h[:, 0, indx[0], indx[1]], axis=(1))
    precip_min = np.min(precip1h[:, 0, indx[0], indx[1]], axis=(1))
    P_bar_max = axma1_3.bar(time_normal, precip_max, color="lightblue", alpha=0.5, width=wd / 4, align="edge", bottom = 0, zorder=10)
    P_bar = axma1_3.bar(time_normal, precip_mean, color="blue", alpha=0.5, width=wd / 4, align="edge", bottom = 0 , zorder=11)
    #P_bar_min = axma1_3.bar(time_normal, precip_min, color="red", alpha=1, width=wd / 4, align="edge", bottom = 0, zorder=12)
    autolabel(P_bar_max, axma1_3, space=1)
    #autolabel(P_bar, axma1_3, space=1)
    axma1_3.set_ylim(bottom=0.001, top = 10)
    axma1_3.set_yticks([])
    axma1_3.legend([T2M_MEAN[0], RH2m_MEAN[0], P_bar[0]],
                   ["2m mean Temp.", "2m mean RH", "1h acc mean/max precip"], loc='upper left').set_zorder(99999)
    xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    axma1.xaxis.set_major_formatter(xfmt)
    #################################
    # P2: Spec Hum. and sample
    #################################
    # specifichum
    q_mean2m = np.mean(dmet.specific_humidity_2m[:, 0, indx[0], indx[1]], axis=(1))
    Q2M_mean = axma2.plot(time_normal, q_mean2m * 1000,
                     zorder=0, color="green", alpha=1, linewidth=3)
    Q2M = axma2.plot(time_normal, dmet.specific_humidity_2m[:, 0,indx[0], indx[1]] * 1000,
                     zorder=0, color="green",alpha=0.7, linewidth=0.2)
    axma2.set_ylabel('Spec. Hum. (g/kg)', color="green")
    axma2.tick_params(axis="y", colors="green")
    axma2.set_ylim(bottom=0, top=1.5)
    # sample
    #sample_ml = get_samplesize(np.mean(dmet.specific_humidity_2m[:, :, indx[0], indx[1]], axis=(1)), density_ml, acc=3)

    axma2_3 = axma2.twinx()
    wd = -0.125  # width of 3h x axis found emprically, should heve better automatic
    sample_ml_mean = np.mean(sample_ml[:, -1, indx[0], indx[1]], axis=(1))
    S_bar = axma2_3.bar(time_normal[0::3], sample_ml_mean[0::3], ls="--", ec="green", color="lightblue",
                       alpha=0.8, width=wd, align="edge")
    autolabel(S_bar, axis=axma2_3, fmt='{:.3f}', space=1)
    axma2_3.set_ylim(bottom=0, top=0.3)
    axma2_3.set_ylabel(' Sample size (g)')
    ## label
    axma2_3.legend([Q2M_mean[0], S_bar[0]], ["2m mean Spec.Hum..", "mean 3h acc samplesize"],
                loc='upper left').set_zorder(99999)

    #################################
    # P3: Wind ans pressure
    #################################
    # wind
    wspeed_gust = np.sqrt(dmet.x_wind_gust_10m[:, 0, indx[0], indx[1]] ** 2 + dmet.y_wind_gust_10m[:, 0, indx[0], indx[1]] ** 2)
    wspeed = np.sqrt(dmet.x_wind_10m[:, 0, indx[0], indx[1]] ** 2 + dmet.y_wind_10m[:, 0, indx[0], indx[1]] ** 2)
    wsg_mean = np.mean(wspeed_gust, axis=(1))
    ws_mean = np.mean(wspeed, axis=(1))
    WIND_MEAN = axma3.plot(time_normal, ws_mean, zorder=1, color="darkmagenta",linewidth=3, alpha=1)
    WIND = axma3.plot(time_normal, wspeed, zorder=1, color="darkmagenta", linewidth=0.2, alpha=0.7)
    GUST_MEAN = axma3.plot(time_normal, wsg_mean, zorder=0, color="magenta",linewidth=3, alpha=1)
    GUST = axma3.plot(time_normal, wspeed_gust, zorder=0, color="magenta", linewidth=0.2, alpha=0.7)

    #axma3.quiver(time_normal, ws_mean, dmet.x_wind_10m[:, 0, jindx, iindx] / wspeed,
    #            dmet.y_wind_10m[:, 0, jindx, iindx] / wspeed, scale=80, zorder=2)
    #
    #axm3.quiver(time_normal, wspeed_gust, dmet.x_wind_gust_10m[:, 0, jindx, iindx] / wspeed_gust,
    #            dmet.y_wind_gust_10m[:, 0, jindx, iindx] / wspeed_gust, scale=80, zorder=2)
    axma3.set_ylabel('wind (m/s)')
    axma3.set_ylim(bottom=0, top=25)
    axma3.tick_params(axis="y", color="darkmagenta")
    # pressure
    #axma3_2 = axma3.twinx()
    #p_mean = np.mean( dmet.surface_air_pressure[:, 0,indx[0], indx[1]], axis=(1))
    #axma3_2.plot(time_normal, p_mean / 100, color="k", linewidth=3, alpha=1)
    #axma3_2.plot(time_normal, dmet.surface_air_pressure[:, 0,indx[0], indx[1]] / 100, color="k", linewidth=0.2, alpha=0.8)
    #axma3_2.set_ylabel(' Surface Pressure (hPa)')
    #axma3_2.set_ylim(bottom=900, top=1050)

    # label
    axma3.legend([GUST_MEAN[0], WIND_MEAN[0]], ["10m wind gust", "10m wind (10min mean)"], loc='upper left').set_zorder(99999)

    #################################
    # P4: FLux of sensible/latent heat
    #################################
    SH_mean = np.mean(dmet_sfx.H[:, indx[0], indx[1]], axis=(1))
    LH_mean = np.mean( dmet_sfx.LE[:, indx[0], indx[1]], axis=(1))
    P_SH_MEAN = axma4.plot(time_normal, SH_mean, zorder=0, color="blue", linewidth=3, alpha=1)
    P_SH = axma4.plot(time_normal, dmet_sfx.H[:, indx[0], indx[1]], zorder=0, color="blue", linewidth=0.2, alpha=0.7)
    P_LH_MEAN = axma4.plot(time_normal, LH_mean, zorder=0, color="orange", linewidth=3, alpha=1)
    P_LH = axma4.plot(time_normal, dmet_sfx.LE[:, indx[0], indx[1]], zorder=0, color="orange", linewidth=0.2, alpha=0.7)
    axma4.set_ylabel('Fluxes (W/m$^2$)')
    axma4.legend( [P_SH_MEAN[0], P_LH_MEAN[0]], ["Sensible Heat Flux", "Latent Heat Flux"], loc='upper left').set_zorder(99999)
    axma4.xaxis.set_major_formatter(xfmt)
    figma1.tight_layout()
    plt.savefig(dirName_b2 + figname_b2 + "_LOC["+sitename+"]" + ".png")
    plt.close()

def main():
    dirName_b0, dirName_b1, dirName_b2,dirName_b3,figname_b0, figname_b1, figname_b2,figname_b3 = setup_directory()

    ############
    ind_list = closest(sites.loc["OldPier"].lat, sites.loc["OldPier"].lon)

    plot_maplocation(ind_list[0:5], dirName_b2, figname_b2)
    ip = 0
    for points in ind_list[0:5]:
        jindx, iindx = points
        plot_meteogram_vertical(jindx, iindx, dirName_b1, figname_b1, ip)
        plot_meteogram(jindx, iindx, dirName_b0, figname_b0, ip)
        ip += 1
    #average plots
    averagesite = ["SEA", "LAND", "PIER", "ALL"]
    for sitename in averagesite:
        if sitename =="SEA":
            indx_sea = np.where(dmet.land_area_fraction[0][0][:][:] == 0)
            meteogram_average( indx_sea ,dirName_b3 ,figname_b3, sitename)
            plot_amaplocation(indx_sea, dirName_b2, figname_b2, sitename)
        if sitename =="LAND":
            indx_land = np.where(dmet.land_area_fraction[0][0][:][:] == 1)
            meteogram_average( indx_land ,dirName_b3 ,figname_b3, sitename)
            plot_amaplocation(indx_land, dirName_b2, figname_b2, sitename)
        if sitename =="PIER":
            ll = np.array([list(item) for item in ind_list[0:5]])
            jindx = ll[:, 0]
            iindx = ll[:, 1]
            meteogram_average( [jindx,iindx] ,dirName_b3 ,figname_b3, sitename)
            plot_amaplocation( [jindx,iindx], dirName_b2, figname_b2, sitename)
        if sitename =="ALL":
            ll = np.array([list(item) for item in ind_list[:]])
            jindx = ll[:, 0]
            iindx = ll[:, 1]
            meteogram_average( [jindx,iindx] ,dirName_b3 ,figname_b3, sitename)
            plot_amaplocation( [jindx,iindx], dirName_b2, figname_b2, sitename)




main()