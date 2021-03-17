from weathervis.config import *
from weathervis.utils import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.check_data import *

from weathervis.calculation import *
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import matplotlib.patheffects as pe
from cartopy.io import shapereader  # For reading shapefiles containg high-resolution coastline.
from copy import deepcopy
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl
from weathervis.checkget_data_handler import *


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def domain_input_handler(dt, model, domain_name, domain_lonlat, file, point_name):
    if domain_name or domain_lonlat or point_name:
        if domain_lonlat:
            print(f"\n####### Setting up domain for coordinates: {domain_lonlat} ##########")
            data_domain = domain(dt, model, file=file, lonlat=domain_lonlat)
        else:
            data_domain = domain(dt, model, file=file)

        if domain_name != None and domain_name in dir(data_domain):
            print(f"\n####### Setting up domain: {domain_name} ##########")
            domain_name = domain_name.strip()
            if re.search("\(\)$", domain_name):
                func = f"data_domain.{domain_name}"
            else:
                func = f"data_domain.{domain_name}()"
            eval(func)
        if point_name and domain_name == None and domain_lonlat ==None:
            data_domain = domain(dt, model, file=file, point_name=point_name)

    else:
        data_domain = None
    return data_domain
def setup_met_directory(modelrun, point_name, point_lonlat):
    projectpath = setup_directory(OUTPUTPATH, "{0}/".format(modelrun))
    #figname = "fc_" + modelrun
    # dirName = projectpath + "result/" + modelrun[0].strftime('%Y/%m/%d/%H/')
    if point_lonlat:
        pname = str(point_lonlat)
        dirName = projectpath  # + "/"+ str(point_lonlat)
        #dirName = projectpath + "meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_lonlat)
    else:
        dirName = projectpath
        pname = str(point_name)  # + "/"+ str(point_name)
        #dirName = projectpath + "meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_name)

    dirName_b1 = dirName
    figname_b1 = "VPMETEOGRAM_" + pname + "_" + modelrun


    dirName_b0 = dirName  # + "met/"
    figname_b0 = "PMETEOGRAM_" + pname + "_" + modelrun

    dirName_b2 = dirName  # + "map/"
    figname_b2 = "map_" + modelrun

    dirName_b3 = dirName  # + "met/"
    figname_b3 = "met_" + modelrun

    if not os.path.exists(dirName_b1):
        os.makedirs(dirName_b1)
        print("Directory ", dirName_b1, " Created ")
    else:
        print("Directory ", dirName_b1, " already exists")
    #if not os.path.exists(dirName_b2):
    #    os.makedirs(dirName_b2)
    #    print("Directory ", dirName_b2, " Created ")
    #else:
    #    print("Directory ", dirName_b2, " already exists")
    return dirName_b0, dirName_b1, dirName_b2, dirName_b3, figname_b0, figname_b1, figname_b2, figname_b3
def nice_vprof_colorbar(CF, ax, lvl=None, ticks=None, label=None, highlight_val=None, highlight_linestyle="k--", extend="both"):
    x0, y0, width, height = 0.75, 0.86, 0.26, 0.13
    axins = inset_axes(ax, width='80%', height='23%',
                        bbox_to_anchor=(x0, y0, width, height),  # (x0, y0, width, height)
                        bbox_transform=ax.transAxes,
                        loc="upper center")
    cbar = plt.colorbar(CF, extend=extend, cax=axins, orientation="horizontal", ticks=ticks, format='%.1f')
    ax.add_patch(plt.Rectangle((x0, y0), width, height, fc=[1, 1, 1, 0.7],
                                 transform=ax.transAxes, zorder=1000))

    if ticks is not None:
        cbar.ax.xaxis.set_tick_params(pad=-0.5) #all ticks closer attatched to bar

    if highlight_val is not None:
        #only for equally spaced lvls now. Maybe use diff later to correct for irregularies
        #diff = [t[i + 1] - t[i] for i in range(len(t) - 1)] #list of diff between element
        len = lvl[-1] - lvl[0] #17
        us = highlight_val - lvl[0]
        loc_ticks = us/float(len)  #0.5625, 0,975862068965517

        cbar.ax.plot([loc_ticks]*2, [0, 1], highlight_linestyle) #additional contour on plot

    if label is not None:
        cbar.set_label( label, labelpad=-0.5 )
    return cbar


class VERT_MET():
    def __init__(self, model, date, steps, data=None, domain_name=None, domain_lonlat=None, legend=None, info=None,
                 num_point=None, point_name=None, point_lonlat=None):
        self.model = model
        self.date = date
        self.steps = steps
        self.data = data
        self.domain_name = domain_name
        self.domain_lonlat = domain_lonlat
        self.num_point = num_point
        self.point_name = point_name
        self.point_lonlat = point_lonlat
        self.param_pl = []
        self.param_ml = ["air_temperature_ml", "specific_humidity_ml", "x_wind_ml", "y_wind_ml",
                          "cloud_area_fraction_ml"]
        self.param_sfc = ["surface_air_pressure", "air_pressure_at_sea_level", "air_temperature_0m","atmosphere_boundary_layer_thickness","surface_geopotential"]
        self.param_sfx = []
        self.param = self.param_ml + self.param_pl + self.param_sfc + self.param_sfx
        self.p_level = None
        self.m_level = None
        self.mbrs = None
        self.url = None
        self.point_lonlat = point_lonlat
        self.num_point = num_point
        date = str(date)

    def retrieve_handler(self):

        print("\n######## Checking if your request is possible ############")
        self.param = self.param_pl + self.param_ml + self.param_sfc + self.param_sfx
        dmet,data_domain,bad_param = checkget_data_handler(all_param=self.param, date=self.date, model=self.model, step=self.steps,
                                     p_level=self.p_level, m_level=self.m_level,mbrs=self.mbrs,
                                     domain_name=self.domain_name, domain_lonlat=self.domain_lonlat, point_name=self.point_name)

        self.dmet = dmet
        self.data_domain = data_domain
        print("DATA RETRIEVED")

        return dmet, data_domain,bad_param
    def calculations(self):
        self.dmet.p = ml2pl(self.dmet.ap, self.dmet.b, self.dmet.surface_air_pressure)
        print(np.shape(self.dmet.ap))

        self.dmet.u,self.dmet.v = xwind2uwind(self.dmet.x_wind_ml, self.dmet.y_wind_ml, self.dmet.alpha)
        print("test1")
        self.dmet.velocity = wind_speed(self.dmet.x_wind_ml, self.dmet.y_wind_ml)
        print("test2")
        self.dmet.heighttoreturn = ml2alt_gl(air_temperature_ml=self.dmet.air_temperature_ml,
                                           specific_humidity_ml=self.dmet.specific_humidity_ml, ap=self.dmet.ap,
                                           b=self.dmet.b,
                                           surface_air_pressure=self.dmet.surface_air_pressure)
        print("test3")
        self.dmet.dtdz = lapserate(self.dmet.air_temperature_ml, self.dmet.heighttoreturn, self.dmet.air_temperature_0m)
        print("test4")
        self.dmet.time_normal = timestamp2utc(self.dmet.time)
        print("test5")
        self.dmet.theta = potential_temperatur(self.dmet.air_temperature_ml, self.dmet.p)
        print("test6")
        self.dmet.altfrom_pref = pl2alt_sl(self.dmet.surface_geopotential, self.dmet.air_temperature_ml, self.dmet.specific_humidity_ml, self.dmet.p*100)

        #self.dmet.reference_pressure=np.full(np.shape(self.dmet.surface_air_pressure),1013.)
        #print(self.dmet.reference_pressure)
        #self.dmet.verticalref = ml2alt_sl(surface_geopotential=self.dmet.surface_geopotential, air_temperature_ml=self.dmet.air_temperature_ml,
        #                                   specific_humidity_ml=self.dmet.specific_humidity_ml, ap=self.dmet.ap,
        #                                   b=self.dmet.b,
        #                                   surface_air_pressure=self.dmet.reference_pressure)

    def points(self):
        point_lonlat = self.point_lonlat; dmet = self.dmet; num_point = self.num_point
        print("find nearest")
        print("#####################################################################")

        if point_lonlat:
            ind_list = nearest_neighbour(point_lonlat[0], point_lonlat[1], dmet.longitude, dmet.latitude, num_point)
        else:
            sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
            ind_list = nearest_neighbour(sites.loc[self.point_name].lon, sites.loc[self.point_name].lat, dmet.longitude,
                                         dmet.latitude, num_point)
            point_lonlat = [sites.loc[self.point_name].lon, sites.loc[self.point_name].lat]
        poi = ind_list[0:num_point]
        return poi

    def vertical_met(self,jindx, iindx, dirName_b1, figname_b1, ip, p_top = 500):
        dmet = self.dmet
        # Point and calc
        p_p = dmet.p[:, :, jindx, iindx]  # / 100
        #refp_p = dmet.altfrom_pref[:, :, jindx, iindx]  # / 100
        Rd = 287.06  # [J/kg K] Gas constant for dry air
        g = 9.80665
        #self.dmet.reference_pressure=np.full(np.shape(self.dmet.surface_air_pressure),1013.)
        #Tv = virtual_temp(dmet.air_temperature_ml, dmet.specific_humidity_ml)
        #h_ref = (Rd * Tv[:, :, jindx, iindx] / g) * np.log(p_p / 101300.)
        #print(p_p)
        #def func_h_ref(p_p):
        #    h_ref = ( Rd *Tv[:, :, jindx, iindx]/g )*np.log(p_p/1013.)
        #    return h_ref
        #def func_p_ref(h_ref):
        #    p_ref = 1013.*np.exp(h_ref * g / ( Rd *Tv[:, :, jindx, iindx]))
        #    return p_ref
        #p_p =x
        q_p = dmet.specific_humidity_ml[:, :, jindx, iindx]  # * 1000
        temp_p = dmet.air_temperature_ml[:, :, jindx, iindx]  # - 273.15
        ur_p = dmet.u[:, :, jindx, iindx]
        vr_p = dmet.v[:, :, jindx, iindx]
        vel_p = dmet.velocity[:, :, jindx, iindx]
        rh_p = relative_humidity(temp_p, q_p, p_p )

        #MASK
        temp_p = np.ma.array(temp_p, mask=p_p < p_top)
        ur_p = np.ma.array(ur_p, mask=p_p < p_top)
        vr_p = np.ma.array(vr_p, mask=p_p < p_top)
        vel_p = np.ma.array(vel_p, mask=p_p < p_top)
        q_p = np.ma.array(q_p, mask=p_p < p_top)
        dtdz_p = np.ma.array(dmet.dtdz[:, :, jindx, iindx], mask=p_p < p_top)
        areafrac_cloud = np.ma.array(dmet.cloud_area_fraction_ml[:, :, jindx, iindx], mask=p_p < p_top)

        #UNIT
        areafrac_cloud=areafrac_cloud*100
        p_p = p_p/100
        q_p = q_p*1000
        temp_p = temp_p - 273.15

        print("\n###########################\n"
              "\nINITIALISING PLOTTING: meteogram_vertical \n"
              "\n###########################\n")
        #INITIALISING
        figm1, (axm1, axm2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), sharex=False)
        plt.subplots_adjust(wspace=0.001)
        levels = range(len(dmet.hybrid))
        lx, tx = np.meshgrid(levels, dmet.time_normal[:])

        #################################
        # P1: RH with lapserate and BLheight
        #################################
        # Ground color gray
        axm1.fill_between(dmet.time_normal[:], dmet.surface_air_pressure[:, 0, jindx, iindx] / 100, \
                                  dmet.air_pressure_at_sea_level[:, 0, jindx, iindx] / 100, color="gray")
        # spec humidity
        cmap = cm.get_cmap('gnuplot2_r')  # BrBu  BrYlBu
        lvl = np.linspace(np.min(q_p),np.max(q_p), 20)
        CF_Q = axm1.contourf(tx, p_p, q_p, levels=lvl, cmap=cmap,extend="both", zorder=1)
        ticks = np.array([lvl[0], lvl[5], lvl[10], lvl[15], lvl[-1]])

        cbar = nice_vprof_colorbar(CF=CF_Q, ax=axm1,ticks=ticks, lvl=lvl, label = 'Spec. Hum. [g/kg]')

        axm1.contour(tx, p_p, q_p, linestyles="dashed",
                     levels=ticks[1:-1], colors="white", zorder=2, alpha=0.8)

        axm1.invert_yaxis()
        axm1.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 600)

        #################################
        # P2: Potential temp with wind
        #################################
        # Ground in gray
        axm2.fill_between(dmet.time_normal[:], dmet.surface_air_pressure[:, 0, jindx, iindx] / 100, \
                                  dmet.air_pressure_at_sea_level[:, 0, jindx, iindx] / 100, color="gray")
        cmap = plt.cm.RdYlBu_r  # plt.cm.jet RdYlBu
        skip = (slice(None, None, 2), slice(None, None, 2))

        axm2.barbs(tx[skip][skip], p_p[skip][skip], ur_p[skip][skip] * 1.943844,
                   vr_p[skip][skip] * 1.943844, length=7, zorder=1000, sizes=dict(emptybarb=0.25, spacing=0.15, height=0.4))

        lvl = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        CF_WS = axm2.contourf(tx, p_p, vel_p, cmap=cmap, alpha=0.8, levels=lvl, extend="both",
                              zorder=1)
        axm2.contour(tx, p_p, vel_p, levels=[13.], linestyles="dashed", colors="black",
                      alpha=0.8, zorder=1)


        ticks = np.array([3, 13, 20])
        cbar = nice_vprof_colorbar(CF=CF_WS, ax=axm2,ticks=ticks, lvl=lvl, label = 'Wind Speed [m/s]', highlight_val = [13] )

        # potential temp.
        #lvl = np.linspace(np.min(dmet.theta[:, :, jindx, iindx]), np.max(dmet.theta[:, :, jindx, iindx]), 200)
        lvl = np.arange(np.min(dmet.theta[:, :, jindx, iindx]), np.max(dmet.theta[:, :, jindx, iindx]), 2)

        CS = axm2.contour(tx, p_p, dmet.theta[:, :, jindx, iindx], colors="black", levels=lvl,
                          zorder=2)
        axm2.clabel(CS, [*CS.levels[2:5:1], *CS.levels[5:10:2], *CS.levels[15:20:5]], inline=True,
                    fmt='$\Theta$ = %1.0fK')  # '%1.0fK')
        # label
        axm2.legend([CS], ["Pot. Temp."], loc='upper left').set_zorder(99999)
        axm2.invert_yaxis()
        axm2.set_ylabel("Pressure [hPa]")
        axm2.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 600)

        #################################
        # SET ADJUSTMENTS ON AXIS
        #################################
        xfmt_maj = mdates.DateFormatter('%d.%m')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_min = mdates.DateFormatter('%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

        axm2.xaxis.set_major_locator(mdates.DayLocator())
        axm2.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axm2.xaxis.set_major_formatter(xfmt_maj)
        axm2.xaxis.set_minor_formatter(xfmt_min)
        axm1.xaxis.set_major_locator(mdates.DayLocator())
        axm1.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axm1.xaxis.set_major_formatter(xfmt_maj)
        axm1.xaxis.set_minor_formatter(xfmt_min)

        axm1.xaxis.grid(True, which="major", linewidth=2)
        axm1.xaxis.grid(True, which="minor", linestyle="--")
        axm2.xaxis.grid(True, which="major", linewidth=2)
        axm2.xaxis.grid(True, which="minor", linestyle="--")
        axm2.tick_params(axis="x", which="major", pad=12)

        #figm1.tight_layout()
        #ymin, ymax = axm1.get_ylim()  refp_p
        #axi = axm1.twinx()
        #secax = axm1.secondary_yaxis('right', functions=(func_h_ref, func_p_ref))
        #axi.set_ylim( np.min(h_ref.squeeze()), np.max(h_ref.squeeze()) )
        #axi.plot([], [])
        #axi.plot(tx, h_ref)

        #axi.set_ylim(axm1.get_ylim())
        #axi.plot(tx,self.dmet.verticalref.squeeze())
        #ax2.plot([],[])

        axm1.text(0, 1, "{0}_VPMET_{1}_{2}".format(self.model,self.point_name, dt), ha='left', va='bottom', transform=axm1.transAxes, color='black')
        axm2.text(0, 1, "{0}_VPMET_{1}_{2}".format(self.model,self.point_name, dt), ha='left', va='bottom', transform=axm2.transAxes, color='black')

        plt.savefig(dirName_b1 + figname_b1 + "_op2"+ ".png", bbox_inches = "tight", dpi = 200)

        #plt.savefig(dirName_b1 + figname_b1 + "_LOC" + str(ip) +
        #           "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[jindx, iindx],
        #                                         dmet.latitude[jindx, iindx]) + ".png")
        plt.clf()
        plt.close()

        # INITIALISING
        figm2, (axm1, axm2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), sharex=False)
        plt.subplots_adjust(wspace=0.001)
        levels = range(len(dmet.hybrid))
        lx, tx = np.meshgrid(levels, dmet.time_normal[:])

        #################################
        # P1: RH with lapserate and BLheight
        #################################
        # Ground color gray
        axm1.fill_between(dmet.time_normal[:], dmet.surface_air_pressure[:, 0, jindx, iindx] / 100, \
                          dmet.air_pressure_at_sea_level[:, 0, jindx, iindx] / 100, color="gray")
        cmap = cm.get_cmap('RdYlBu_r')  # BrBu  BrYlBu cool bwr RdYlBu_r
        lvl1 = np.linspace(-15, -9.8, 5)
        lvl2 = np.linspace(-6.5, -0.5, 5)
        lvl3 = np.linspace(0, 10, 5)
        lvl = np.append(lvl1, lvl2)
        lvl = np.append(lvl, lvl3)
        ticks = np.array([-9.8,-6.5, -3, 0, 3, 6])
        norm = mpl.colors.DivergingNorm(vmin=-10., vcenter=0., vmax=6)
        CF = axm1.pcolormesh(tx, p_p, dtdz_p, cmap=cmap, zorder=1,norm=norm) #dtdz_p
        cbar = nice_vprof_colorbar(CF=CF, ax=axm1,ticks=ticks, label = 'Lapse. rate. [C/km]' )
        #relative humidity
        CS = axm1.contour(tx, p_p, rh_p, zorder=2, levels = np.arange(0,100,10),colors="green")  #Purples BrBu  BrYlBu cool bwr RdYlBu_r
        axm1.clabel(CS, inline=True, fmt='%1.0f')  # '%1.0fK')
        #cloud
        Cfrac = axm1.contourf(tx, p_p, areafrac_cloud,hatches=['--','---'], colors="none", alpha = 0.0,levels= [1,50,100],zlevel=100)
        artists, labels = Cfrac.legend_elements()
        cfrac_leg = axm1.legend(artists, ["1-50% Cloud cover"," 50-100% Cloud cover"], handleheight=2, loc='upper left')

        #adjust axis
        axm1.invert_yaxis()
        axm1.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 600)
        #axm1.legend(Cfrac[0], "1%-50% Cloud cover")

        #TEMP
        cmap = cm.get_cmap('twilight_shifted')  # BrBu  BrYlBu
        norm = mpl.colors.DivergingNorm(vmin=-30., vcenter=0., vmax=20)
        CF_2 = axm2.pcolormesh(tx, p_p, temp_p, zorder=1, cmap=cmap, norm=norm) #dtdz_p
        cbar = nice_vprof_colorbar(CF=CF_2, ax=axm2, label = 'Temp. [K]', extend="both")

        #RH
        C = axm2.contour(tx, p_p, rh_p, zorder=2, levels = np.arange(0,100,10),colors="green") #dtdz_p
        axm2.clabel(C, inline=True, fmt='%1.0f')  # '%1.0fK')
        #BLH
        #BL = dmet.BLH[:, 0, jindx, iindx]
        #print(np.shape(dmet.BLH)) #(11, 1, 11, 14)
        #print(np.shape(BL)) #(11,)
        #print(tx)

        #h_gl = dmet.atmosphere_boundary_layer_thickness[:, 0, jindx, iindx]
        #h_sl = h_gl + dmet.surface_geopotential[:, 0, jindx, iindx] / 9.08
        #

        #h = np.repeat(h_sl, repeats=len(dmet.hybrid), axis=0).reshape(
        #    np.shape(dmet.specific_humidity_ml[:, :, jindx, iindx]))
        #BL = point_alt_sl2pres_old(jindx, iindx,
        #                   h,
        #                   dmet.geotoreturn, dmet.t_v_level, p, dmet.surface_air_pressure,
        #                   dmet.surface_geopotential)
        #BL = point_alt_sl2pres(jindx, iindx, h, geotoreturn, t_v_level, p, dmet.surface_air_pressure, dmet.surface_geopotential)
        #BL = BL/100
        #P_BL = axm2.plot(tx, BL, "X-", color="black", linewidth=3, zorder=1000)  # (67, 1, 949, 739)
        #axm2.annotate("BLH", (tx[1], BL[1]), color="black", zorder=1000)
        #axm2.clabel(C, inline=True, fmt='%1.0f')  # '%1.0fK')

        #adjust axis
        axm2.invert_yaxis()
        axm2.set_ylim(dmet.air_pressure_at_sea_level[:, 0, jindx, iindx].max() / 100, 600)

        #axis
        xfmt_maj = mdates.DateFormatter('%d.%m')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_min = mdates.DateFormatter('%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

        axm2.xaxis.set_major_locator(mdates.DayLocator())
        axm2.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axm2.xaxis.set_major_formatter(xfmt_maj)
        axm2.xaxis.set_minor_formatter(xfmt_min)
        axm1.xaxis.set_major_locator(mdates.DayLocator())
        axm1.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axm1.xaxis.set_major_formatter(xfmt_maj)
        axm1.xaxis.set_minor_formatter(xfmt_min)

        axm1.xaxis.grid(True, which="major", linewidth=2)
        axm1.xaxis.grid(True, which="minor", linestyle="--")
        axm2.xaxis.grid(True, which="major", linewidth=2)
        axm2.xaxis.grid(True, which="minor", linestyle="--")
        axm2.tick_params(axis="x", which="major", pad=12)

        #figm2.tight_layout()
        print(" SAVEIIING")
        print(dirName_b1 + figname_b1 + "_op1_"+ ".png")
        axm1.text(0, 1, "{0}_VPMET_{1}_{2}".format(self.model,self.point_name, dt), ha='left', va='bottom', transform=axm1.transAxes, color='black')
        axm2.text(0, 1, "{0}_VPMET_{1}_{2}".format(self.model,self.point_name, dt), ha='left', va='bottom', transform=axm2.transAxes, color='black')
        plt.savefig(dirName_b1 + figname_b1 + "_op1"+ ".png", bbox_inches = "tight", dpi = 200)

        plt.clf()
        plt.close()
        print("DONE SAVE")

def handle_input():
    param_ML = ["air_temperature_ml", "specific_humidity_ml", "x_wind_ml", "y_wind_ml", "cloud_area_fraction_ml"]
    param_SFC = ["surface_air_pressure"]


    dmet_ml.p = ml2pl(dmet_ml.ap, dmet_ml.b, dmet_ml.surface_air_pressure)
    dmet_ml.heighttoreturnhalf = ml2alt_gl(air_temperature_ml=dmet_ml.air_temperature_ml,
                                           specific_humidity_ml=dmet_ml.specific_humidity_ml, ap=dmet_ml.ap,
                                           b=dmet_ml.b,
                                           surface_air_pressure=dmet.surface_air_pressure, inputlevel="half",
                                           returnlevel="full")
    dmet_ml.dtdz = lapserate( dmet_ml.air_temperature_ml, dmet_ml.heighttoreturn, dmet.air_temperature_0m )

if __name__ == "__main__":
    import argparse
    def none_or_str(value):
        if value == 'None':
            return None
        return value
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
    parser.add_argument("--steps", default=[0, 10], nargs="+", type=int,
                        help="forecast times example --steps 0 3 gives time 0 to 3")
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument("--domain_name", default=None, help="see domain.py", type=none_or_str)
    parser.add_argument("--domain_lonlat", default=None, nargs="+", type=float, help="lonmin lonmax latmin latmax")
    parser.add_argument("--point_name", default=None, help="see sites.csv")
    parser.add_argument("--point_lonlat", default=None, nargs="+", type=float, help="lon lat")
    parser.add_argument("--point_num", default=1, type=int)
    parser.add_argument("--plot", default="all", help="Display legend")
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()

    for dt in args.datetime:
        dirName_b0, dirName_b1, dirName_b2, dirName_b3, figname_b0, figname_b1, figname_b2, figname_b3 = setup_met_directory(
            dt, args.point_name, args.point_lonlat)

        VM = VERT_MET(date=dt, steps=args.steps, model=args.model, domain_name=args.domain_name,
                      domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, num_point=args.point_num,
                      point_lonlat=args.point_lonlat, point_name=args.point_name)

        VM.retrieve_handler()
        VM.calculations()

        points = VM.points()
        ip=0
        for po in points:
            jindx, iindx = po
            VM.vertical_met(jindx, iindx, dirName_b1, figname_b1, ip)
            ip += 1
