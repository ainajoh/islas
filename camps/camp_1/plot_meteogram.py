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

path = os.path.abspath("islas/camps/camp_1")

param_ML = ["air_temperature_ml","specific_humidity_ml"]
param_SFC = ["surface_air_pressure", "air_pressure_at_sea_level", "surface_geopotential","atmosphere_boundary_layer_thickness"]
param_sfx = None

print("set domain area and points")
data_domain = DOMAIN()
data_domain.KingsBay_Z1()
sites = pd.read_csv("sites.csv", sep=";", header=0, index_col=0)
ZeppelinObservatory = sites.loc["ZeppelinObservatory"]

print("start retrieve")
########################################
#RETRIEVE
dmet = DATA(data_domain=data_domain, param_SFC = param_SFC, param_ML=param_ML, fctime=[0,2])
dmet.retrieve()
dmet_sfx = DATA(data_domain=data_domain, param_sfx = param_sfx, fctime=[0,2], type = "sfx")
dmet_sfx.retrieve()
print("retrieve done")
#########################################
#CALCULATE VARIABLES
print("start calc")
time_normal = timestamp2utc(dmet.time)
p = ml2pl(dmet.ap, dmet.b, dmet.surface_air_pressure)
theta = potential_temperatur(dmet.air_temperature_ml, p)
print(theta)
specific_humidity_ml= dmet.specific_humidity_ml*1000.
geotoreturn = ml2alt_sl( p, dmet.surface_geopotential, dmet.air_temperature_ml, dmet.specific_humidity_ml )
heighttoreturn = ml2alt_gl(p, dmet.air_temperature_ml, dmet.specific_humidity_ml)
t_v_level = virtual_temp(dmet.air_temperature_ml, dmet.specific_humidity_ml)
dtdz = lapserate(dmet.air_temperature_ml, heighttoreturn)
print("calc done")

def point_calculation(jindx, iindx):
    height = np.full(np.shape(specific_humidity_ml[:, :, jindx, iindx]), float(ZeppelinObservatory.height))
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


def meteogram_vertical(jindx, iindx):
    ZeppelinObservatory_height_pl, BL = point_calculation(jindx, iindx)

    p[:, :, jindx, iindx] = p[:, :, jindx, iindx]/100
    ZeppelinObservatory_height_pl = ZeppelinObservatory_height_pl/100
    BL = BL/100

    print("inside meteogram_vertical")
    fig2, ax2 = plt.subplots( figsize = ( 12, 4 ) )
    levels = range( len( dmet.hybrid ) )
    lx, tx = np.meshgrid( levels, time_normal[:] )

    #POTTEMP
    lvl = np.linspace(np.min(theta[:, :, jindx, iindx]), np.max(theta[:, :, jindx, iindx]), 60)
    CS = ax2.contour( tx, p[:, :, jindx, iindx], theta[:, :, jindx, iindx], colors = "black", levels=lvl, zorder=900, label = "pottwmp")

    #Q
    cmap= cm.get_cmap( 'gnuplot2_r' ) #BrBu  BrYlBu
    lvl = np.linspace( np.min( specific_humidity_ml[:, :, jindx, iindx] ), np.max( specific_humidity_ml[:, :, jindx, iindx] ),20)
    CF_Q = ax2.contourf( tx, p[:, :, jindx, iindx], specific_humidity_ml[:, :, jindx, iindx], levels=lvl, cmap = cmap)
    axins1 = inset_axes(ax2, width = '80%',height = '15%',
                        bbox_to_anchor=(0.88,0.84,0.12,0.15),
                        bbox_transform=ax2.transAxes,
                        loc="upper center")
    #plot ground.
    paths = ax2.fill_between(time_normal[:], dmet.surface_air_pressure[:, 0, jindx, iindx]/100, \
                             dmet.air_pressure_at_sea_level[:, 0, jindx, iindx]/100, color="gray")
    #HEIGHT OF ZEPPELINER.
    Z = ax2.plot(time_normal[:], ZeppelinObservatory_height_pl, "^", linestyle = ':', color="c", zorder = 1000, alpha = 0.7)
    ax2.annotate("Zeppelin = 479m", (time_normal[1], ZeppelinObservatory_height_pl[1]), color = "c", zorder=1000)


    P_BL= ax2.plot( time_normal[:],BL, "X-", color = "black", linewidth=3, zorder=1000) #(67, 1, 949, 739)
    ax2.annotate("PBLH", (time_normal[1], BL[1]), color="black", zorder=1000)

    #Lapserate.
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

    plt.clabel(CL,inline=False, fmt='%.1f' + r"K/km")
    plt.clabel(CS, [*CS.levels[3:5:1], *CS.levels[5:10:2], *CS.levels[15:20:5]], inline=False, fmt='$\Theta$ = %1.0fK')#'%1.0fK')
    c = plt.clabel(CS, [CS.levels[2]], fmt='$\Theta$ = %1.0fK')
    ax2.invert_yaxis()

    #ax2.legend( [Z[0],P_BL[0]],['Zeppelin = 479m', 'PBLH'], loc='upper left').set_zorder(99999)
    custom_lines = [Line2D([0], [0], color="r", lw=2),
                    Line2D([0], [0], color="green", lw=2),
                    Line2D([0], [0], color="gray", lw=2)]

    ax2.legend( [Z[0],P_BL[0], custom_lines[0], custom_lines[1],custom_lines[2]], ['Zeppelin = 479m', 'PBLH', 'Unstable', 'Very Stable', 'Stable'], loc='upper left').set_zorder(99999)
    #ax2.legend(custom_lines, [r'$\Gamma_e > \Gamma_d$', r'$\Gamma_e > 0 $', 'Hot'], loc='upper left').set_zorder(99999)


    ax2.set_ylabel("Pressure [hPa]")
    xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
    ax2.xaxis.set_major_formatter(xfmt)  # Setting xaxis to this format
    plt.ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max()/100, 100 )



    # For visualization purposes we mark the bounding box by a rectangle
    ax2.add_patch(plt.Rectangle((0.88, 0.84), 0.12, 0.15, fc=[1, 1, 1, 0.7],
                               transform=ax2.transAxes, zorder = 1000))
    #f = ax2.patches.extend([plt.Rectangle((0.863, 0.89), 0.092, 0.042,
    #                                      fill=False, color='k', alpha=0.8, zorder=9,
    #                                      transform=fig2.transFigure, figure=fig2)])

    ticks = np.linspace(np.min(CF_Q.levels), np.max(CF_Q.levels), 4)
    #ticks = np.linspace(0.2, 4, 4)
    cbar = plt.colorbar(CF_Q , extend = "both",  cax=axins1, orientation="horizontal", ticks=ticks, format='%.1f')
    cbar.ax.xaxis.set_tick_params(pad=-0.5)
    cbar.set_label('Spec. Hum. [g/kg]', labelpad=-0.5)
    #axins1.xaxis.set_ticks_position("bottom") ticks=range(lvl.min(), lvl.max(),5), boundaries=bounds_knots

    fig2.tight_layout()
    plt.rcParams["axes.axisbelow"] = True

    plt.show()

meteogram_vertical(0, 0)
