from domain import *
from get_data import *
from calculate_newdata import *
import os
import matplotlib.pyplot as plt                 #For basic plotting in python
import matplotlib
import matplotlib.dates as mdates
import matplotlib.cm as cm                      #For colors on map
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib as mpl


import cartopy.crs as ccrs                      #For setting up a map
from cartopy.io import shapereader              #For reading shapefiles containg high-resolution coastline.

path = os.path.abspath("islas/camps/camp_1")

param_ML = ["air_temperature_ml","specific_humidity_ml", "x_wind_ml", "y_wind_ml"]
param_SFC =["air_temperature_2m", "surface_air_pressure", "air_pressure_at_sea_level",
            "surface_geopotential","atmosphere_boundary_layer_thickness", "relative_humidity_2m",
            "x_wind_gust_10m", "y_wind_gust_10m", "x_wind_10m", "y_wind_10m",
            "integral_of_surface_downward_sensible_heat_flux_wrt_time",
            "integral_of_surface_downward_latent_heat_flux_wrt_time", "specific_humidity_2m",
            "precipitation_amount_acc"]
param_sfx = ["SST","H","LE"]

print("\n###########################\n"
      "\n# SET DOMAIN               #\n"
      "\n###########################\n")
data_domain = DOMAIN()
data_domain.KingsBay_Z1()
sites = pd.read_csv("../sites.csv", sep=";", header=0, index_col=0)
ZeppelinObservatory = sites.loc["ZeppelinObservatory"]
OldPier = sites.loc["OldPier"]

print("\n###########################\n"
      "\n# RETRIEVE DATA            #\n"
      "\n###########################\n")
########################################
#RETRIEVE
dmet = DATA(data_domain=data_domain, param_SFC = param_SFC, param_ML=param_ML, fctime=[0,66], modelrun="2020020500")
dmet.retrieve()
dmet_sfx = DATA(data_domain=data_domain, param_sfx = param_sfx, fctime=[0,66], type = "sfx",  modelrun="2020020500")
dmet_sfx.retrieve()


print("\n###########################\n"
      "\n# CALCULATE and set DATA          #\n"
      "\n###########################\n")

time_normal = timestamp2utc(dmet.time)
modelrun = timestamp2utc([dmet.forecast_reference_time])

precip1h = precip_acc(dmet.precipitation_amount_acc, acc=1)
p = ml2pl(dmet.ap, dmet.b, dmet.surface_air_pressure)
theta = potential_temperatur(dmet.air_temperature_ml, p)
specific_humidity_ml= dmet.specific_humidity_ml*1000.
geotoreturn = ml2alt_sl( p, dmet.surface_geopotential, dmet.air_temperature_ml, dmet.specific_humidity_ml )
heighttoreturn = ml2alt_gl(p, dmet.air_temperature_ml, dmet.specific_humidity_ml)
t_v_level = virtual_temp(dmet.air_temperature_ml, dmet.specific_humidity_ml)
dtdz = lapserate(dmet.air_temperature_ml, heighttoreturn)
SH = dmet.integral_of_surface_downward_sensible_heat_flux_wrt_time
LH = dmet.integral_of_surface_downward_latent_heat_flux_wrt_time

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
    BL = point_alt_sl2pres(jindx, iindx, h, geotoreturn, t_v_level, p, dmet.surface_air_pressure, dmet.surface_geopotential)

    return ZeppelinObservatory_height_pl, BL


def meteogram_vertical(jindx, iindx, ax2):
    ZeppelinObservatory_height_pl, BL = point_calculation(jindx, iindx)

    #Correct unit for proper display
    p[:, :, jindx, iindx] = p[:, :, jindx, iindx]/100
    ZeppelinObservatory_height_pl = ZeppelinObservatory_height_pl/100
    BL = BL/100
    #################################

    print("\n###########################\n"
          "\nINITIALISING PLOTTING: meteogram_vertical \n"
          "\n###########################\n")

    levels = range( len( dmet.hybrid ) )
    lx, tx = np.meshgrid( levels, time_normal[:] )


    #################################
    #P1:POTENTIAL TEMP
    #################################
    lvl = np.linspace(np.min(theta[:, :, jindx, iindx]), np.max(theta[:, :, jindx, iindx]), 60)
    CS = ax2.contour( tx, p[:, :, jindx, iindx], theta[:, :, jindx, iindx], colors = "black", levels=lvl, zorder=900, label = "pottwmp")
    #################################
    #P2:HUMIDITY
    #################################
    cmap= cm.get_cmap( 'gnuplot2_r' ) #BrBu  BrYlBu
    lvl = np.linspace( np.min( specific_humidity_ml[:, :, jindx, iindx] ), np.max( specific_humidity_ml[:, :, jindx, iindx] ),20)
    CF_Q = ax2.contourf( tx, p[:, :, jindx, iindx], specific_humidity_ml[:, :, jindx, iindx], levels=lvl, cmap = cmap)
    axins1 = inset_axes(ax2, width = '80%',height = '15%',
                        bbox_to_anchor=(0.88,0.84,0.12,0.15),
                        bbox_transform=ax2.transAxes,
                        loc="upper center")
    #################################
    #P3:DFINE WHAT IS GROUND: diff between pressure at sea level and surface pressure
    #################################
    paths = ax2.fill_between(time_normal[:], dmet.surface_air_pressure[:, 0, jindx, iindx]/100, \
                             dmet.air_pressure_at_sea_level[:, 0, jindx, iindx]/100, color="gray")
    #################################
    #P4:HEIGHT OF ZEPPELINER.
    #################################
    Z = ax2.plot(time_normal[:], ZeppelinObservatory_height_pl, "^", linestyle = ':', color="c", zorder = 1000, alpha = 0.7)
    ax2.annotate("Zeppelin = 479m", (time_normal[1], ZeppelinObservatory_height_pl[1]), color = "c", zorder=1000)

    #################################
    #P5:HEIGHT OF BL
    #################################
    P_BL= ax2.plot( time_normal[:],BL, "X-", color = "black", linewidth=3, zorder=1000) #(67, 1, 949, 739)
    ax2.annotate("BLH", (time_normal[1], BL[1]), color="black", zorder=1000)

    #################################
    #P6:LAPSERATE
    #################################
    lvl1 = np.linspace(-15, -9.8, 5 )
    lvl2 = np.linspace(-6.5,-0.5, 5 )
    lvl3 = np.linspace(0, 10, 5)
    lvl = np.append(lvl1,lvl2)
    lvl = np.append(lvl,lvl3)
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    CL=ax2.contour(tx, p[:, :, jindx, iindx], dtdz[:, :, jindx, iindx],levels = lvl, colors="gray", linewidth=1, zorder=100,)
    unstable = CL.collections[0:5]
    plt.setp(unstable, color="r",  zorder=100)
    stableinv = CL.collections[10:15]
    plt.setp(stableinv, color="green", zorder=100)

    #################################
    #P7:WIND
    #################################
    n = 7
    C = np.sqrt(dmet.x_wind_ml[:, ::n, jindx, iindx] ** 2 + dmet.y_wind_ml[:, ::n, jindx, iindx] ** 2) * 1.943844
    cmap = plt.cm.jet
    bounds_ms = [3., 6., 9., 12.]
    bounds_knots = [x * 1.943844 for x in bounds_ms]
    norm = mpl.colors.BoundaryNorm(bounds_knots, cmap.N)
    img = ax2.barbs(tx[:, ::n], p[:, ::n, jindx, iindx], dmet.x_wind_ml[:, ::n, jindx, iindx] * 1.943844,
                    dmet.y_wind_ml[:, ::n, jindx, iindx] * 1.943844, C, cmap=cmap, norm=norm, length=5)

    #################################
    #ADJUSTMENTS AND LABELS
    #################################
    plt.clabel(CL,inline=False, fmt='%.1f' + r"K/km")
    plt.clabel(CS, [*CS.levels[3:5:1], *CS.levels[5:10:2], *CS.levels[15:20:5]], inline=False, fmt='$\Theta$ = %1.0fK')#'%1.0fK')
    c = plt.clabel(CS, [CS.levels[2]], fmt='$\Theta$ = %1.0fK')
    ax2.invert_yaxis()
    custom_lines = [Line2D([0], [0], color="r", lw=2),
                    Line2D([0], [0], color="green", lw=2),
                    Line2D([0], [0], color="gray", lw=2)]

    ax2.legend( [Z[0],P_BL[0], custom_lines[0], custom_lines[1],custom_lines[2]], ['Zeppelin = 479m', 'PBLH', 'Unstable', 'Very Stable', 'Stable'], loc='upper left').set_zorder(99999)
    ax2.set_ylabel("Pressure [hPa]")
    xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    ax2.xaxis.set_major_formatter(xfmt)  # Setting xaxis to this format
    ax2.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max()/100, 100 )
    ax2.add_patch(plt.Rectangle((0.88, 0.84), 0.12, 0.15, fc=[1, 1, 1, 0.7],
                               transform=ax2.transAxes, zorder = 1000))
    ticks = np.linspace(np.min(CF_Q.levels), np.max(CF_Q.levels), 4)
    cbar = plt.colorbar(CF_Q , extend = "both",  cax=axins1, orientation="horizontal", ticks=ticks, format='%.1f')
    cbar.ax.xaxis.set_tick_params(pad=-0.5)
    cbar.set_label('Spec. Hum. [g/kg]', labelpad=-0.5)
    plt.rcParams["axes.axisbelow"] = True

#meteogram_vertical(0, 0)
def meteogram(jindx, iindx, axm1, axm2, axm4):

    xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

    #################################
    # P2: TEMP and RH and PRECIP
    #################################
    def autolabel(rects): #for the precip. Got from e
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects[::2]:
            height = rect.get_height()
            axm1_3.annotate('{0:.1f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    axm1_2 = axm1.twinx()
    axm1_3 = axm1.twinx()
    axm1_4 = axm1.twinx()
    axm1.plot( time_normal, dmet.air_temperature_2m[ :, 0, jindx,iindx] - 273.15, color = "red")
    axm1.set_ylabel('2m Temp ($^\circ$C)', color = "red")
    axm1.tick_params(axis="y", colors="red")
    P_bar = axm1_3.bar(time_normal, precip1h[:, 0, jindx, iindx], color = "blue", alpha = 0.8, width=0.03)
    axm1_2.plot(time_normal, dmet.relative_humidity_2m[:, 0, jindx, iindx]*100, color = "green", alpha = 0.8)
    autolabel(P_bar)
    axm1_4.plot( time_normal, dmet.surface_air_pressure[:, 0, jindx, iindx]/100, color = "k")
    axm1_2.set_ylabel(' 2m Rel. Hum. (%)', color = "green")
    axm1_2.tick_params(axis = "y", colors = "green" )
    axm1_3.set_yticks([])
    axm1_3.set_xticks([])
    axm1_4.tick_params(axis="y", direction="in", pad=-22)
    axm1_4.set_ylabel("Surface pressure [hPa]", labelpad = -33)
    axm1.xaxis.set_major_formatter(xfmt)

    #################################
    # P2: WIND and SENSIBLE HEAT
    #################################
    axm2_2 = axm2.twinx()
    wspeed_gust = np.sqrt( dmet.x_wind_gust_10m[:, 0, jindx, iindx] **2 + dmet.y_wind_gust_10m[:, 0, jindx, iindx] **2)
    wspeed = np.sqrt( dmet.x_wind_10m[:, 0, jindx, iindx] ** 2 + dmet.y_wind_10m[:, 0, jindx, iindx] ** 2 )
    axm2.plot(time_normal,wspeed_gust,zorder=0)
    axm2.plot(time_normal, wspeed, zorder=1)
    Q = axm2.quiver(time_normal, wspeed, dmet.x_wind_10m[:, 0, jindx, iindx]/wspeed,dmet.y_wind_10m[:, 0, jindx, iindx]/wspeed, scale = 80, zorder=2)
    Q_gust = axm2.quiver(time_normal, wspeed_gust, dmet.x_wind_gust_10m[:, 0, jindx, iindx]/wspeed_gust, dmet.y_wind_gust_10m[:, 0, jindx, iindx]/wspeed_gust, scale = 80, zorder=2)
    axm2.set_ylabel('10m Wind/Gust (m/s)')
    axm2_2.plot(time_normal, dmet.specific_humidity_2m[:, 0, jindx, iindx]*1000, zorder=0, color="green", alpha = 0.8)
    #axm2_2.plot(time_normal, dmet.specific_humidity_ml[:, -1, jindx, iindx]*1000, zorder=0, color="green", alpha = 0.5)
    axm2_2.set_ylabel('Spec. Hum. (g/kg)', color = "green")

    axm2.xaxis.set_major_formatter(xfmt)

    #################################
    # P3: FLUXES
    #################################
    P_SH = axm4.plot(time_normal, dmet_sfx.H[:, jindx, iindx], zorder=0, color = "blue")
    P_LH = axm4.plot(time_normal, dmet_sfx.LE[:, jindx, iindx], zorder=0, color = "orange")
    axm4.set_ylabel('Fluxes (W/m$^2$)')
    axm4.xaxis.set_major_formatter(xfmt)
    axm4.legend([P_SH[0], P_LH[0]], ["Sensible Heat Flux", "Latent Heat Flux"], loc='upper left')
    #fig5: Cloud cover. (maybe merge with fig3 later.)
    return axm1, axm2, axm4

def background_map(lonlat):
    projection = ccrs.UTM( 33 ) #Map projection you want
    crs_latlon = ccrs.PlateCarree() #Coordinates used to plot points on map
    ax = plt.axes( projection=projection )
    ax.set_extent( lonlat )
    shp = shapereader.Reader('/Users/ainajoh/Data/ISAS/shape_files/NP_S100_SHP/S100_Land_f.shp')
    shpkvote=shapereader.Reader('/Users/ainajoh/Data/ISAS/shape_files/NP_S100_SHP/S100_Koter_l.shp')
    gl = ax.gridlines(crs=crs_latlon, draw_labels=False, linewidth=1, color='gray', alpha=0.3, linestyle='--')
    #Coastline
    ax.add_geometries( shp.geometries(), projection, facecolor='gray', edgecolor='black', zorder=2 )
    ax.add_geometries( shp.geometries(), projection,  edgecolor='black', zorder=100, facecolor='None')

    #Heightcontours
    ax.add_geometries( shpkvote.geometries(), projection, facecolor="None", edgecolor='white', alpha = 0.1,  zorder=2 )
    #ax.coastlines()

    points = plt.plot(dmet.longitude, dmet.latitude, marker='.', markersize=5.0, markeredgewidth=4,
                      markerfacecolor='black', markeredgecolor='black', zorder=100, transform=crs_latlon,
                      linestyle='None')

    # PLOT COORDINATES FOR OLDPIER
    plt.plot(OldPier.lon, OldPier.lat, marker='o', markersize=5.0, markeredgewidth=2.5,
             markerfacecolor='blue', markeredgecolor='blue', zorder=100, transform=crs_latlon)

    return projection, crs_latlon

def plot_site():

    points = plt.plot(lons_gridPointDomain, lats_gridPointDomain, marker='.', markersize=5.0, markeredgewidth=4,
                      markerfacecolor='black', markeredgecolor='black', zorder=6, transform=crs_latlon,
                      linestyle='None')

    #PLOT GRIDPOINT OF INTEREST
    points = plt.plot(lons_gridPointDomain,lats_gridPointDomain, marker='.', markersize=5.0, markeredgewidth=4,
                     markerfacecolor='black', markeredgecolor='black', zorder=6, transform=crs_latlon, linestyle = 'None')

    return projection, crs_latlon

def mapwithdata(point, dirName_b2,figname_b2):
    figm2, ax = plt.subplots(figsize=(10, 9))

    map_domain = DOMAIN()
    map_domain.KingsBay_Z0()
    lonlat = np.array(map_domain.lonlat)

    projection, crs_latlon = background_map(lonlat)
    map_data_domain = DOMAIN()
    map_data_domain.KingsBay_Z0()
    plt.savefig(dirName_b2 + figname_b2 + ".png")
    plt.close()

    param_SFC = ["x_wind_10m", "y_wind_10m", "air_temperature_2m", ]  # ["air_temperature_2m", "surface_air_pressure", "air_pressure_at_sea_level",

    dmap = DATA(data_domain=map_data_domain, param_SFC=param_SFC, fctime=[0, 66], modelrun="2020020500")
    dmap.retrieve()

    for t in range(0,np.shape(dmap.time)[0]):
        figm2, ax = plt.subplots(figsize=(10, 9))
        projection, crs_latlon = background_map(lonlat)

        cmap = cm.get_cmap('twilight_shifted')
        CFW= plt.contourf(dmap.longitude,dmap.latitude,dmap.air_temperature_2m[t,0,:,:]-273.15, transform=crs_latlon, zorder = 10, alpha = 0.9, cmap=cmap)
        plt.Rectangle((0.88, 0.84), 0.12, 0.15, fc=[1, 1, 1, 0.7])
        #ticks = np.linspace(np.min(dmap.air_temperature_2m), np.max(dmap.air_temperature_2m), 4)
        cbar = plt.colorbar(CFW, extend="both", format='%.0f', fraction=0.02, pad=0.01)
        cbar.set_label('2m Temp. [C]')

        wspeed = np.sqrt(dmap.x_wind_10m[t,0,:,:] ** 2 + dmap.y_wind_10m[t,0,:,:] ** 2)
        plt.barbs(dmap.longitude,dmap.latitude,dmap.x_wind_10m[t,0,:,:],dmap.y_wind_10m[t,0,:,:], transform=crs_latlon, color = "black", zorder = 20)
        figname_b2_1 = figname_b2 + "+"+ str(t)
        plt.savefig(dirName_b2+figname_b2_1+".png")

        ip = 0
        for p in point:

            points = plt.plot(dmet.longitude[p], dmet.latitude[p], marker='.', markersize=6.0, markeredgewidth=4,
                              markerfacecolor='red', markeredgecolor='red', zorder=100, transform=crs_latlon,
                              linestyle='None')
            figname_b2_2 = figname_b2_1 + "_LOC"+str(ip)+"["+"{0:.2f}_{1:.2f}]".format(dmet.longitude[p], dmet.latitude[p])
            plt.savefig(dirName_b2+figname_b2_2+".png")

            points = plt.plot(dmet.longitude[p], dmet.latitude[p], marker='.', markersize=6.0, markeredgewidth=4,
                        markerfacecolor='black', markeredgecolor='black', zorder=100, transform=crs_latlon,
                        linestyle='None')
            ip+=1
        plt.close()



def each_point(point, dirName_b1,figname_b1,ip):#figsize=(12, 14)figsize=(5, 7)
    jindx, iindx = point
    figm1, (ax2, axm1, axm2, axm4) = plt.subplots(nrows=4, ncols=1, figsize=(12, 14), sharex=True)
    meteogram_vertical(jindx, iindx, ax2)
    meteogram(jindx, iindx, axm1, axm2, axm4)
    figm1.tight_layout()
    plt.savefig(dirName_b1 + figname_b1 + "_LOC"+str(ip)+ "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[point], dmet.latitude[point]) + ".png")
    #plt.show()
    plt.close()



def closest(lat,lon):
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
    projectpath = "/Users/ainajoh/Documents/Result/ISLAS/camp_1/"
    figname = "fc_"+ modelrun[0].strftime('%Y%m%d%H')
    #dirName = projectpath + "result/" + modelrun[0].strftime('%Y/%m/%d/%H/')
    dirName = projectpath + "result/" + "fc_"+modelrun[0].strftime('%Y%m%d/')

    dirName_b1 = dirName + "meteograms/"
    figname_b1 = "met_"+ figname

    dirName_b2 = dirName + "maps/"
    figname_b2 = "map_"+figname

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
    return dirName_b1, dirName_b2, figname_b1, figname_b2

def meteogram_average(indx):
    lona = lon[indx[0],indx[1]]
    lata = lat[indx[0],indx[1]]

    figma1, (axma1,axma2,axma4) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True )
    # figm1.suptitle('test title', fontsize=20)
    #axma1 = ax[0,0]
    #axma2 = ax[1,0]
    #axma4 = ax[2,0]

    tmp_mean = np.mean( air_temperature_2m[:, 0, indx[0], indx[1] ], axis=(1) )

    axma1_2 = axma1.twinx()
    axma1.plot(time_normal, tmp_mean[:] - 273.15, color = "black")
    axma1.plot(time_normal, air_temperature_2m[:, 0, indx[0], indx[1] ] - 273.15, color = "blue")
    axma1.set_ylabel('2m Temp ($^\circ$C)', color = "blue")
    axma1_2.plot(time_normal, relative_humidity_2m[:, 0,indx[0], indx[1]] * 100, color="green")
    axma1_2.set_ylabel(' 2m Rel. Hum. (%)', color = "green")
    xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    axma1.xaxis.set_major_formatter(xfmt)

    ##fig2: Wind speed: wind_gust, wind_direction.
    # TODO: When u have height in meters calc: Find Wind at say 80 m height/ height of mountain in areas.
    #figma2, axma2 = plt.subplots(figsize=(12, 4))
    # figm2.suptitle('test title', fontsize=20)
    # axm2_2 = axm1.twinx()
    wspeed_gust = np.sqrt(x_wind_gust_10m[:, 0, indx[0], indx[1]] ** 2 + y_wind_gust_10m[:, 0, indx[0], indx[1]] ** 2)
    wspeed = np.sqrt(x_wind_10m[:, 0, indx[0], indx[1]] ** 2 + y_wind_10m[:, 0, indx[0], indx[1]] ** 2)
    plot_wind_gust=axma2.plot(time_normal, wspeed_gust, zorder=0, color = "gray")
    plot_wind=axma2.plot(time_normal, wspeed, zorder=1, color = "black")
    wspeed_mean = np.mean(wspeed, axis = (1))
    wspeed_meanx = np.mean(x_wind_10m[:, 0, indx[0], indx[1]], axis = (1))
    wspeed_meany = np.mean(y_wind_10m[:, 0, indx[0], indx[1]], axis = (1))
    w = axma2.quiver(time_normal, wspeed_mean, wspeed_meanx[:] / wspeed_mean, wspeed_meany[:] / wspeed_mean, scale=70, zorder=2, color = "r")
    wspeed_g_mean = np.mean(wspeed_gust, axis=(1))
    wspeed_g_meanx = np.mean(x_wind_gust_10m[:, 0, indx[0], indx[1]], axis=(1))
    wspeed_g_meany = np.mean(y_wind_gust_10m[:, 0, indx[0], indx[1]], axis=(1))
    w_g = axma2.quiver(time_normal, wspeed_g_mean, wspeed_g_meanx[:] / wspeed_g_mean, wspeed_g_meany[:] / wspeed_g_mean, scale=70, zorder=2, color ="r")
    axma2.set_ylabel('10m Wind/Gust (m/s)')
    #axma2.legend([plot_wind_gust[0],plot_wind[0]],['Wind gust', 'Wind'])
    axma2.xaxis.set_major_formatter(xfmt)
    # fig4: Fluxes radiation : latent, sensible, SW down, LW down, LW_up
    #figma4, axma4 = plt.subplots(figsize=(12, 4))
    plotsh = axma4.plot(time_normal, SH[:, 0, indx[0], indx[1]], zorder=0, color = "yellow")
    plotlh = axma4.plot(time_normal, LH[:, 0, indx[0], indx[1]], zorder=0, color ="red")
    axma4.legend([plotsh[0],plotlh[0]],['Sensible heat flux', 'Latent heat flux'])
    axma4.set_ylabel('Fluxes (Ws/m$^2$)')
    axma4.xaxis.set_major_formatter(xfmt)


def main():
    dirName_b1, dirName_b2, figname_b1, figname_b2 = setup_directory()

    ############
    ind_list = closest(sites.loc["OldPier"].lat, sites.loc["OldPier"].lon)
    mapwithdata(ind_list[0:5], dirName_b2,figname_b2 )
    ip=0
    for points in ind_list[0:5]:
        each_point( points, dirName_b1,figname_b1, ip )
        ip+=1
    ##########
    indx_land = np.where(land_area_fraction[0][0][:][:] == 1)
    indx_sea = np.where(land_area_fraction[0][0][:][:] == 1)







     main()