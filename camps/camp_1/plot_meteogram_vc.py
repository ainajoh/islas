from domain import *
from get_data import *
import os
import matplotlib.pyplot as plt                 #For basic plotting in python
import datetime as dt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.cm as cm                      #For colors on map
import math
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
path = os.path.abspath("islas/camps/camp_1")

param_ML = ["air_temperature_ml","x_wind_ml","y_wind_ml","specific_humidity_ml"]
param_SFC = ["atmosphere_boundary_layer_thickness","air_temperature_2m", "surface_air_pressure","relative_humidity_2m",\
             "x_wind_10m", "y_wind_10m","x_wind_gust_10m", "y_wind_gust_10m",\
             "integral_of_surface_downward_sensible_heat_flux_wrt_time",\
             "integral_of_surface_downward_latent_heat_flux_wrt_time","land_area_fraction"]

param_sfx = ["SST","H","LE"]

param_ML = ["air_temperature_ml","specific_humidity_ml"]
param_SFC = ["surface_air_pressure", "air_pressure_at_sea_level", "surface_geopotential","atmosphere_boundary_layer_thickness"]

print("domain")
data_domain = DOMAIN()
data_domain.KingsBay_Z1()
print("retrieve")
dmet = DATA(data_domain=data_domain, param_SFC = param_SFC, param_ML=param_ML, fctime=[0,60])
dmet.retrieve()

dmet_sfx = DATA(data_domain=data_domain, param_sfx = param_sfx, fctime=[0,60], type = "sfx")
dmet_sfx.retrieve()
#########################################
#CALCULATE VARIABLES
#print("calculating var time")
time_normal = [dt.datetime.utcfromtimestamp(x) for x in dmet.time]
#print("calculating var BL")
#BL_p = dmet.surface_air_pressure*np.exp(-0.00012*dmet.atmosphere_boundary_layer_thickness)/100
#Hybrid to pressure levels
print("calculating var p from hyb")
p = np.zeros(shape = np.shape(dmet.air_temperature_ml))

for k in range(0,len(dmet.hybrid)): # Outside for loop? p = [ac/100 + bc * psc for ac, bc, psc in zip(ap,b, ps[:,0,:,:])]
    p[:,k,:,:] = dmet.ap[k]/100. + dmet.b[k] * dmet.surface_air_pressure[:,0,:,:]/100.
print("calculating pot temp")
p0=100000/100
theta = dmet.air_temperature_ml *(p0/p)**0.286
specific_humidity_ml = dmet.specific_humidity_ml*1000.

def calculategeoh():
    #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    heighttoreturn = np.zeros(shape=np.shape(dmet.air_temperature_ml))
    geotoreturn    = np.zeros(shape=np.shape(dmet.air_temperature_ml))
    t_v_level      = np.zeros(shape=np.shape(dmet.air_temperature_ml))
    dp_level      = np.zeros(shape=np.shape(dmet.air_temperature_ml))
    dlogP_level = np.zeros(shape=np.shape(dmet.air_temperature_ml))
    #dtheta_level = np.zeros(shape=np.shape(dmet.air_temperature_ml))

    z_h = 0

    levelSize = len(dmet.hybrid)
    levels = np.arange(0, len(dmet.hybrid))
    p_low = p[:, levelSize-1, :, :]*100  #Pa lowest modellecel is 64
    reversedlevels = np.full(levels.shape[0], -999, np.int32)
    for iLev in list(reversed(range(levels.shape[0]))):
        reversedlevels[levels.shape[0] - 1 - iLev] = levels[iLev]

    for k in reversedlevels:
        p_top = p[:, k-1, :, :]*100 #pa
        t_v_level[:, k, :, :] = dmet.air_temperature_ml[:, k, :, :] * (1. + 0.609133 * dmet.specific_humidity_ml[:, k, :, :])

        if k == 0: # top of atmos, last loop round
            dlogP = np.log(p_low / 0.1)
            dP = np.nan
            #dtheta = np.nan
            alpha = math.log(2)  # 0.3 why?
        else:
            dlogP = np.log( np.divide(p_low, p_top) )
            dP = p_low - p_top
            #dtheta = theta[:, k, :, :] - theta[:, k-1, :, :]
            alpha = 1. - ((p_top/ dP) * dlogP)


        dp_level[:,k,:,:] = dP
        dlogP_level[:,k,:,:] = dlogP
        #dtheta_level[:,k,:,:] = dtheta
        Rd = 287.06
        TRd = t_v_level[:, k, :, :] * Rd
        z_f = z_h + (TRd * alpha)

        heighttoreturn[:, k, :, :] = z_f / 9.80665 #m

        geotoreturn[:, k, :, :] = z_f + dmet.surface_geopotential[:, 0, :, :]
        z_h = z_h + (TRd * dlogP)
        p_low = p_top
    return geotoreturn/9.80665, heighttoreturn, t_v_level,dp_level,dlogP_level

geotoreturn, heighttoreturn, t_v_level, dp_level, dlogP_level = calculategeoh()

def lapserate():
    dt_levels = np.full(np.shape(specific_humidity_ml[:, :, :, :]), np.nan)
    dz_levels = np.full(np.shape(specific_humidity_ml[:, :, :, :]), np.nan)
    dtheta_levels = np.full(np.shape(specific_humidity_ml[:, :, :, :]), np.nan)

    dtdz = np.full(np.shape(specific_humidity_ml[:, :, :, :]), np.nan)
    #dz = 1000 #m
    step=5
    for k in range(0, len(dmet.hybrid)-step):  # Outside for loop? p = [ac/100 + bc * psc for ac, bc, psc in zip(ap,b, ps[:,0,:,:])]
        k_next=k+step
        dt_levels[:, k, :, :] = dmet.air_temperature_ml[:, k, :, :] - dmet.air_temperature_ml[:, k_next, :, :] #over -under
        dtheta_levels[:, k, :, :] = theta[:, k, :, :] - theta[:, k_next, :, :] #over -under
        dz_levels[:, k, :, :] = heighttoreturn[:, k, :, :] - heighttoreturn[:, k_next, :, :] #over -under

    #floc1[0::2] - floc1[1::2]
    #dtheta_levels = theta[:,:-1,:,: ] - theta[:, 1:,:,:]
    #dt_levels[:,0:-1,:,:] = dmet.air_temperature_ml[:, :-1,:,:] - dmet.air_temperature_ml[:, 1:,:,:]

    #dz_levels[:,0:-1,:,:] = heighttoreturn[:, :-1,:,:] - heighttoreturn[:, 1:,:,:]
    dtdz[:,:,:,:] = np.divide(dt_levels,dz_levels)*1000 #/km

    print(np.shape(dtdz))
    return dtdz
    #dt/dz per km
dtdz = lapserate()

def alt2pres(jindx, iindx, h):
    g = 9.807 #m/s**2
    R_d = 287
    #print(geotoreturn[:,:,jindx, iindx].min())
    #idx_tk = np.where( ( geotoreturn[:,:,jindx, iindx] <= h ) )
    idx_tk = np.argmax( ( geotoreturn[:,:,jindx, iindx] <= h[:] ), axis=1 )
    tv = t_v_level[:,:,jindx, iindx]
    dp = dp_level[:,:,jindx, iindx]
    dlogP = dlogP_level[:,:,jindx, iindx]
    for t in range(0, np.shape(tv)[0]):#0,1,2
        tv[t,0:idx_tk[t]] = np.nan
        dp[t, 0:idx_tk[t]] = np.nan
        dlogP[t, 0:idx_tk[t]] = np.nan
    tvdlogP = np.multiply(tv,dlogP)

    T_vmean = np.divide( np.nansum( tvdlogP, axis=1), np.nansum(dlogP,axis=1))
    H = R_d * T_vmean/g #scale height
    DP = dmet.air_pressure_at_sea_level[:,0,jindx, iindx] - dmet.surface_air_pressure[:,-1,jindx, iindx]
    h_gl = h[:,0]-dmet.surface_geopotential[:,0,jindx, iindx]/g #convert to height over surface.

    p_point = dmet.surface_air_pressure[:,-1,jindx, iindx] * np.exp(-( np.array( h_gl )/H ) )

    return p_point

def point(jindx, iindx):
    sites = pd.read_csv("sites.csv", sep=";", header=0, index_col=0)
    height = float(sites.loc["ZeppelinObservatory"].height)
    #height_point = np.full(np.shape(dmet.air_temperature_ml)[0], height)
    height_point = np.full( np.shape( specific_humidity_ml[:, :, jindx, iindx] ), height)

    #https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_20200201T12Z.nc?SST[0:1:0][0:1:0][0:1:0]
    #dz = height_point- dmet.surface_geopotential[:,0,jindx, iindx]
    p_point = alt2pres(jindx, iindx, h = height_point)
    return p_point


def meteogram_vertical(jindx, iindx):

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
    p_point = point(jindx, iindx)/100
    Z = ax2.plot(time_normal[:], p_point, "^", linestyle = ':', color="c", zorder = 1000, alpha = 0.7)
    ax2.annotate("Zeppelin = 479m", (time_normal[1], p_point[1]), color = "c", zorder=1000)


    #BL height.
    h_gl =  dmet.atmosphere_boundary_layer_thickness[:, 0, jindx, iindx]
    h_sl = h_gl + dmet.surface_geopotential[:, 0, jindx, iindx]/9.08
    #
    h = np.repeat( h_sl, \
                   repeats=len(dmet.hybrid), axis=0).reshape( np.shape(specific_humidity_ml[:, :, jindx, iindx]) )
    BL = alt2pres(jindx, iindx, h = h )/100
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
    plt.clabel(CS, [*CS.levels[3:5:1], *CS.levels[5:10:2], *CS.levels[15:20:5]], inline=False, fmt='%1.0fK')
    c = plt.clabel(CS, [CS.levels[2]], fmt='$\Theta$ = %1.0fK')
    ax2.invert_yaxis()

    #ax2.legend( [Z[0],P_BL[0]],['Zeppelin = 479m', 'PBLH'], loc='upper left').set_zorder(99999)
    #ax2.legend( [CS, Z[0],P_BL[0]],['pot','Zeppelin = 479m', 'PBLH'], loc='upper left').set_zorder(99999)
    #ax2.legend( CS,['pot'], loc='upper left').set_zorder(99999)
    #ax2.legend(CS, loc='upper left').set_zorder(99999)

    custom_lines = [Line2D([0], [0], color="r", lw=2),
                    Line2D([0], [0], color="green", lw=2),
                    Line2D([0], [0], color="gray", lw=2)]

    ax2.legend(custom_lines, ['Unstable', 'Very Stable', 'Stable'], loc='upper left').set_zorder(99999)
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
