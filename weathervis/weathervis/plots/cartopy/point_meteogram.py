from weathervis.config import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.calculation import *
from weathervis.checkget_data_handler import *

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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def domain_input_handler(dt, model, domain_name, domain_lonlat, file):
    if domain_name or domain_lonlat:
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
    else:
        data_domain = None
    return data_domain
def setup_directory(modelrun, point_name, point_lonlat):
    projectpath = "../../../output/"
    figname = "fc_" + modelrun
    # dirName = projectpath + "result/" + modelrun[0].strftime('%Y/%m/%d/%H/')
    if point_lonlat:
        dirName = projectpath + "result/" + "fc_" + modelrun[:-2] + "/" + str(point_lonlat)
    else:
        dirName = projectpath + "result/" + "fc_" + modelrun[:-2] + "/" + str(point_name)

    dirName_b1 = dirName + "met/"
    figname_b1 = "vmet_" + figname

    dirName_b0 = dirName + "met/"
    figname_b0 = "met_" + figname

    dirName_b2 = dirName + "map/"
    figname_b2 = "map_" + figname

    dirName_b3 = dirName + "met/"
    figname_b3 = "met_" + figname

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
    return dirName_b0, dirName_b1, dirName_b2, dirName_b3, figname_b0, figname_b1, figname_b2, figname_b3

class PMET():
    def __init__(self, model, date, steps, data = None, domain_name = None, domain_lonlat=None,legend=None,info=None,
                 num_point=None,point_name=None, point_lonlat=None):
        print("pmet")
        self.model =model
        self.date=date
        self.steps=steps
        self.data=data
        self.domain_name = domain_name
        self.domain_lonlat = domain_lonlat
        self.num_point = num_point
        self.point_name = point_name
        self.point_lonlat = point_lonlat
        self.param_pl  = []
        self.param_ml  = ["air_temperature_ml", "specific_humidity_ml"]
        self.param_sfc = ["surface_air_pressure","air_pressure_at_sea_level","air_temperature_0m","air_temperature_2m",
                          "relative_humidity_2m", "x_wind_gust_10m","y_wind_gust_10m","x_wind_10m","y_wind_10m",
                          "specific_humidity_2m", "precipitation_amount_acc","convective_cloud_area_fraction",
                          "cloud_area_fraction","high_type_cloud_area_fraction","medium_type_cloud_area_fraction",
                          "low_type_cloud_area_fraction","rainfall_amount","snowfall_amount","graupelfall_amount",
                          "land_area_fraction"]
        self.param_sfx = ["H", "LE"]
        self.param = self.param_ml + self.param_pl + self.param_sfc + self.param_sfx
        self.p_level = None
        self.m_level = [64]
        self.mbrs = None
        self.url = None
        self.point_lonlat = point_lonlat
        self.num_point = num_point
        date = str(date)

    def retrieve_handler_old(self):
        dmet_sfx = None;
        dmet_ml = None;
        dmet_pl = None;
        dmet_sfc = None;
        dmet = None
        split = False
        print("\n######## Checking if your request is possibel ############")
        try:
            self.param = self.param_pl + self.param_ml + self.param_sfc
            check_all = check_data(date=self.date, model=self.model, param=self.param, step=self.steps, p_level=self.p_level, m_level=self.m_level,
                                   mbrs=self.mbrs)
        except ValueError:
            split = True
            try:
                print("--------> Splitting up your request to find match ############")
                check_pl = check_data(date=self.date, model=self.model, param=self.param_pl, step=self.steps, p_level=self.p_level,
                                      mbrs=self.mbrs) if self.param_pl is not None else None
                check_sfc = check_data(date=self.date, model=self.model, param=self.param_sfc, step=self.steps,
                                       mbrs=self.mbrs) if self.param_sfc is not None else None
                check_ml = check_data(date=self.date, model=self.model, param=self.param_ml, step=self.steps, m_level=self.m_level,
                                      mbrs=self.mbrs) if self.param_ml is not None else None
            except ValueError:
                print("!!!!! Sorry this plot is not availbale for this date. Try with another datetime !!!!!")
                sys.exit(1)
                # break
        print("--------> Found match for your request ############")
        if self.param_sfx:
            print("\n######## Checking if your sfx request is possibel ############")
            try:
                check_sfx = check_data(date=self.date, model=self.model, param=self.param_sfx, step=self.steps)
            except ValueError:
                param_sfx = ["SFX_SST", "SFX_H", "SFX_LE", "SFX_TS"]
                try:
                    check_sfx = check_data(date=self.date, model=self.model, param=self.param_sfx, step=self.steps)
                except ValueError:
                    print(
                        "!!!!! Missing surfex data. Sorry this plot is not availbale for this date. Try with another datetime !!!!!")
                    sys.exit(1)
                    # break
            print("--------> Found match for your sfx request ############")
            print(check_sfx.file)
            print("\n######## Retriving sfx data ############")
            file_sfx = check_sfx.file.loc[0]
            data_domain = domain_input_handler(self.date, self.model, self.domain_name, self.domain_lonlat, file_sfx)
            dmet_sfx = get_data(model=self.model, data_domain=data_domain, param=self.param_sfx, file=file_sfx, step=self.steps,
                                date=self.date)
            dmet_sfx.retrieve()

        if not split:
            file_all = check_all.file.loc[0]
            data_domain = domain_input_handler(self.date, self.model, self.domain_name, self.domain_lonlat,file_all)
            dmet = get_data(model=self.model, data_domain=data_domain, param=self.param, file=file_all,
                            step=self.steps, date=self.date, p_level=self.p_level, m_level=self.m_level, mbrs=self.mbrs)
            print("\n######## Retriving data ############")
            print(f"--------> from: {dmet.url} ")
            dmet.retrieve()
            dmet_ml = dmet  # two names for same value, no copying done.
            dmet_pl = dmet
            dmet_sfc = dmet

        else:
            # get sfc level data
            file_sfc = check_sfc.file.loc[0]
            fixed_var = np.array(["ap", "b", "ap2", "b2", "hybrid", "hybrid2"])
            data_domain = domain_input_handler(self.date, self.model, self.domain_name, self.domain_lonlat, file_sfc)
            dmet_sfc = get_data(model=self.model, param=self.param_sfc, file=file_sfc, step=self.steps, date=dt,
                                data_domain=data_domain, mbrs=self.mbrs)
            for val in fixed_var: dmet_sfc.param = np.delete(dmet_sfc.param, np.where(dmet_sfc.param==val)) if val in dmet_sfc.param else dmet_sfc.param

            file_pl = check_pl.file.loc[0]
            # data_domain = domain_input_handler(dt, model, domain_name, domain_lonlat, file_pl)
            dmet_pl = get_data(model=self.model, param=self.param_pl, file=file_pl, step=self.steps, date=dt,
                               data_domain=data_domain, p_level=self.p_level, mbrs=self.mbrs)
            for val in fixed_var: dmet_pl.param = np.delete(dmet_pl.param, np.where(dmet_pl.param==val)) if val in dmet_pl.param else dmet_pl.param

            file_ml = check_ml.file.loc[0]
            # data_domain = domain_input_handler(dt, model, domain_name, domain_lonlat, file_ml)
            dmet_ml = get_data(model=self.model, param=self.param_ml, file=file_ml, step=self.steps, date=dt,
                               data_domain=data_domain, m_level=self.m_level, mbrs=self.mbrs)

            print("\n######## Retriving data ############")
            print(f"--------> from: {dmet_pl.url} ")
            dmet_pl.retrieve()
            print("\n######## Retriving data ############")
            print(f"--------> from: {dmet_ml.url} ")
            dmet_ml.retrieve()
            print("\n######## Retriving data ############")
            print(f"--------> from: {dmet_sfc.url} ")
            dmet_sfc.retrieve()
            dmet = dmet_ml

        if dmet_sfx is not None:
            for pm in dmet_sfx.param:
                setattr(dmet, pm, getattr(dmet_sfx, pm))
                # setattr(dmet, pm, getattr(dmet_sfx, pm))
                # units
        if dmet_pl is not None:
            for pm in dmet_pl.param:
                setattr(dmet, pm, getattr(dmet_pl, pm))
            # if dmet_ml is not None:
            #    for pm in dmet_ml.param:
            #        setattr(dmet, pm, getattr(dmet_ml, pm))
        if dmet_sfc is not None:
            for pm in dmet_sfc.param:
                setattr(dmet, pm, getattr(dmet_sfc, pm))


        self.dmet = dmet
        self.data_domain = data_domain
        #return dmet, data_domain

    def retrieve_handler(self):
        dmet_sfx = None;
        dmet_ml = None;
        dmet_pl = None;
        dmet_sfc = None;
        dmet = None
        split = False
        print("\n######## Checking if your request is possibel ############")
        self.param = self.param_pl + self.param_ml + self.param_sfc + self.param_sfx
        dmet = checkget_data_handler(all_param=self.param, date=self.date, model=self.model, step=self.steps, p_level=self.p_level, m_level=self.m_level,
                                   mbrs=self.mbrs)

        #return dmet, data_domain

    def calculations(self):
        self.dmet.time_normal = timestamp2utc(self.dmet.time)
        self.dmet.precip1h = precip_acc(self.dmet.precipitation_amount_acc, acc=1)
        self.dmet.ug_10m, self.dmet.vg_10m = xwind2uwind(self.dmet.x_wind_gust_10m, self.dmet.y_wind_gust_10m, self.dmet.alpha)
        self.dmet.u_10m, self.dmet.v_10m = xwind2uwind(self.dmet.x_wind_10m, self.dmet.y_wind_10m, self.dmet.alpha)

    def points(self):
        point_lonlat = self.point_lonlat; dmet = self.dmet; num_point = self.num_point
        if point_lonlat:
            ind_list = nearest_neighbour(point_lonlat[0], point_lonlat[1], dmet.longitude, dmet.latitude, num_point)
        else:
            sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
            ind_list = nearest_neighbour(sites.loc[self.point_name].lon, sites.loc[self.point_name].lat, dmet.longitude,
                                         dmet.latitude, num_point)
            point_lonlat = [sites.loc[self.point_name].lon, sites.loc[self.point_name].lat]
        poi = ind_list[0:self.num_point]
        indx_sea = np.where(dmet.land_area_fraction[0][0][:][:] == 0)
        indx_land = np.where(dmet.land_area_fraction[0][0][:][:] == 1)
        indx_alldomain = np.where(dmet.latitude != None)
        ll = np.array([list(item) for item in ind_list[0:num_point]])
        jindx = ll[:, 0]
        iindx = ll[:, 1]
        index_neares = [jindx, iindx]
        return poi, indx_sea, indx_land, indx_alldomain, index_neares

    def plot_meteogram(self,jindx, iindx, dirName_b0, figname_b0, ip):
        dmet=self.dmet
        print("\n###########################\n"
              "\nINITIALISING PLOTTING: meteogram \n"
              "\n###########################\n")
        figm2, (axm1, axm2, axm3, axm4) = plt.subplots(nrows=4, ncols=1, figsize=(12, 14), sharex=True)

        def autolabel(rects, axis, fmt='{0:.1f}', space=2):  # for the precip. Got from e
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
        # temp
        T2M = axm1.plot(dmet.time_normal, dmet.air_temperature_2m[:, -1, jindx, iindx] - 273.15, color="brown")
        TML0 = axm1.plot(dmet.time_normal, dmet.air_temperature_ml[:, -1, jindx, iindx] - 273.15, "--", color="brown")
        axm1.set_ylabel('Temp ($^\circ$C)', color="brown")
        axm1.tick_params(axis="y", colors="brown")
        TS = axm1.plot(dmet.time_normal, dmet.air_temperature_0m[:, -1, jindx, iindx] - 273.15, "-.", color="brown")
        axm1.set_ylabel('Temp ($^\circ$C)', color="brown")
        axm1.tick_params(axis="y", colors="brown")
        min = np.min(
            [dmet.air_temperature_ml[:, -1, jindx, iindx], dmet.air_temperature_0m[:, -1, jindx, iindx]]) - 273.15
        max = np.max(
            [dmet.air_temperature_ml[:, -1, jindx, iindx], dmet.air_temperature_0m[:, -1, jindx, iindx]]) - 273.15

        axm1.set_ylim(bottom=np.floor(min), top=np.ceil(max))

        axm1_2 = axm1.twinx()
        # RH
        axm1_2.plot(dmet.time_normal, dmet.relative_humidity_2m[:, 0, jindx, iindx] * 100, color="seagreen", alpha=0.8)
        axm1_2.set_ylabel(' 2m Rel. Hum. (%)', color="seagreen")
        axm1_2.set_ylim(bottom=0.001, top=100)
        axm1_2.tick_params(axis="y", colors="seagreen")

        # precip
        axm1_3 = axm1.twinx()
        wd = -0.125
        P_bar = axm1_3.bar(dmet.time_normal, dmet.precip1h[:, 0, jindx, iindx], color="blue", alpha=0.8, width=wd / 4,
                           align="edge")
        topidx = 1.
        maxp = np.nanmax(dmet.precip1h[:, 0, jindx, iindx])
        print(maxp)
        if maxp > topidx:
            topidx = round_up(maxp)
        print(topidx)
        axm1_3.set_ylim(bottom=0.001,
                        top=topidx + 0.1 * topidx)  # adding 20% of max value for getting some clearense above bar
        autolabel(P_bar, axm1_3, space=1)
        axm1_3.set_yticks([])
        axm1_3.set_xticks([])

        # label
        axm1_3.legend([T2M[0], TML0[0], TS[0], P_bar[0]], ["2m Temp.", "ml0 Temp.", "0m Temp", "1h acc precip"],
                      loc='upper left').set_zorder(99999)

        #################################
        # P2: SENSIBLE HEAT, sample size
        #################################
        # specifichum
        Q2M = axm2.plot(dmet.time_normal, dmet.specific_humidity_2m[:, 0, jindx, iindx] * 1000, zorder=2, color="green",
                        alpha=0.8)
        QML0 = axm2.plot(dmet.time_normal, dmet.specific_humidity_ml[:, -1, jindx, iindx] * 1000, "--", zorder=2,
                         color="green",
                         alpha=0.8)
        axm2.set_ylabel('Spec. Hum. (g/kg)', color="green")
        axm2.tick_params(axis="y", colors="green")

        maxq = np.max(dmet.specific_humidity_ml[:, -1, jindx, iindx] * 1000)
        topidx = 1.5
        if maxq > 1.5:
            topidx = round_up(maxq, 1)
        axm2.set_ylim(bottom=0, top=topidx + 0.10 * topidx)
        # cloudtypefraclevels.and blh
        # cloud_area_fraction convective_cloud_area_fraction high_type_cloud_area_fraction medium_type_cloud_area_fraction low_type_cloud_area_fraction
        axm2_1 = axm2.twinx()
        # stem

        tot_clf_f = axm2_1.fill_between(dmet.time_normal, 0, dmet.cloud_area_fraction[:, -1, jindx, iindx] * 100,
                                        zorder=1, color="gray", alpha=0.6)
        tot_clf = axm2_1.plot(dmet.time_normal, dmet.cloud_area_fraction[:, -1, jindx, iindx] * 100, zorder=1,
                              color="k")
        tot_patch = mpl.patches.Patch(color='r', alpha=0.5, linewidth=0)

        conv_clf = axm2_1.plot(dmet.time_normal, dmet.convective_cloud_area_fraction[:, -1, jindx, iindx] * 100,
                               zorder=2, color="pink")
        high_clf = axm2_1.plot(dmet.time_normal, dmet.high_type_cloud_area_fraction[:, -1, jindx, iindx] * 100,
                               zorder=2, color="lightblue", marker=r"$C_H$", markersize=12)
        med_clf = axm2_1.plot(dmet.time_normal, dmet.medium_type_cloud_area_fraction[:, -1, jindx, iindx] * 100,
                              zorder=2, color="blue", marker=r"$C_M$", markersize=12)
        low_clf = axm2_1.plot(dmet.time_normal, dmet.low_type_cloud_area_fraction[:, -1, jindx, iindx] * 100, zorder=2,
                              color="red", marker=r"$C_L$", markersize=12)
        axm2_1.set_ylabel('Cloud cover %', color="k")
        axm2_1.tick_params(axis="y", colors="k")
        # label
        axm2_1.legend([Q2M[0], QML0[0], (tot_clf[0], tot_patch), high_clf[0], med_clf[0], low_clf[0], conv_clf[0]],
                      ["2m Spec.Hum.", "ml0 Spec.Hum.", "tot.cloud", "hi. cloud", "med. cloud", "low cloud",
                       "conv. cloud"],
                      loc='upper left').set_zorder(99999)
        # axm2.legend([Q2M[0], QML0[0]], ["2m Spec.Hum.", "ml0 Spec.Hum."],
        #            loc='upper left').set_zorder(99999)

        #################################
        # P3: Wind, pressure.
        #################################
        # wind
        wspeed_gust = np.sqrt(dmet.ug_10m[:, 0, jindx, iindx] ** 2 + dmet.vg_10m[:, 0, jindx, iindx] ** 2)
        wspeed = np.sqrt(dmet.u_10m[:, 0, jindx, iindx] ** 2 + dmet.v_10m[:, 0, jindx, iindx] ** 2)
        GUST = axm3.plot(dmet.time_normal, wspeed_gust, zorder=1, color="magenta")
        WIND = axm3.plot(dmet.time_normal, wspeed, zorder=1, color="darkmagenta")
        axm3.quiver(dmet.time_normal, wspeed, dmet.u_10m[:, 0, jindx, iindx] / wspeed,
                    dmet.v_10m[:, 0, jindx, iindx] / wspeed, scale=80, zorder=2)

        #    autolabel(P_bar, axm1_3, space=1)

        axm3.quiver(dmet.time_normal, wspeed_gust, dmet.ug_10m[:, 0, jindx, iindx] / wspeed_gust,
                    dmet.vg_10m[:, 0, jindx, iindx] / wspeed_gust, scale=80, zorder=2)

        axm3.set_ylabel('wind (m/s)')
        axm3.set_ylim(bottom=0, top=25)
        axm3.tick_params(axis="y", color="darkmagenta")
        # pressure
        axm3_2 = axm3.twinx()
        axm3_2.plot(dmet.time_normal, dmet.surface_air_pressure[:, 0, jindx, iindx] / 100, zorder=1, color="k")
        axm3_2.set_ylabel(' Surface Pressure (hPa)')
        # axm3_2.set_ylim(bottom=900, top=1050)
        # label
        axm3_2.legend([GUST[0], WIND[0]], ["10m wind gust", "10m wind (10min mean)"], loc='upper left').set_zorder(
            99999)

        #################################
        # P3: FLUXES
        #################################
        P_SH = axm4.plot(dmet.time_normal, dmet.H[:, jindx, iindx], zorder=0, color="blue")
        P_LH = axm4.plot(dmet.time_normal, dmet.LE[:, jindx, iindx], zorder=0, color="orange")
        axm4.set_ylabel('Heat Fluxes (W/m$^2$)')
        # rainfall_amount
        axm4_1 = axm4.twinx()
        tot = dmet.rainfall_amount[:, -1, jindx, iindx] + dmet.snowfall_amount[:, -1, jindx,
                                                          iindx] + dmet.graupelfall_amount[:, -1, jindx, iindx]
        rain_frac = ((dmet.rainfall_amount[:, -1, jindx, iindx]) / tot) * 100
        snow_frac = ((dmet.snowfall_amount[:, -1, jindx, iindx]) / tot) * 100
        graupel_frac = ((dmet.graupelfall_amount[:, -1, jindx, iindx]) / tot) * 100

        rain_in = axm4_1.plot(dmet.time_normal, rain_frac, zorder=1, color="red", linestyle='dashed', marker='o')
        snow_in = axm4_1.plot(dmet.time_normal, snow_frac, zorder=1, color="gray", linestyle='dashed', marker='*')
        graupel_in = axm4_1.plot(dmet.time_normal, graupel_frac, zorder=1, color="lightblue", linestyle='dashed',
                                 marker='D')
        # tot =  axm4_1.plot(dmet.time_normal, tot, zorder=1,color="k")
        axm4_1.tick_params(axis="y", colors="k")
        axm4_1.set_ylabel('% of instantaneous precip. type')

        # label

        axm4.legend([P_SH[0], P_LH[0], rain_in[0], snow_in[0], graupel_in[0]],
                    ["Sensible H.Flux", "Latent H.Flux", "Rain", "Snow", "Graupel"],
                    loc='upper left').set_zorder(99999)

        # axm4.legend([P_SH[0], P_LH[0]], ["Sensible Heat Flux", "Latent Heat Flux"], loc='upper left').set_zorder(99999)

        #################################
        # SET ADJUSTMENTS ON AXIS
        #################################
        xfmt = mdates.DateFormatter(
            '%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_maj = mdates.DateFormatter('%d.%m')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_min = mdates.DateFormatter('%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

        axm4.xaxis.set_major_locator(mdates.DayLocator())
        axm4.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axm4.xaxis.set_major_formatter(xfmt_maj)
        axm4.xaxis.set_minor_formatter(xfmt_min)

        axm1.xaxis.grid(True, which="major", linewidth=2)
        axm1.xaxis.grid(True, which="minor", linestyle="--")
        axm2.xaxis.grid(True, which="major", linewidth=2)
        axm2.xaxis.grid(True, which="minor", linestyle="--")
        axm3.xaxis.grid(True, which="major", linewidth=2)
        axm3.xaxis.grid(True, which="minor", linestyle="--")
        axm4.xaxis.grid(True, which="major", linewidth=2)
        axm4.xaxis.grid(True, which="minor", linestyle="--")

        axm4.tick_params(axis="x", which="major", pad=12)
        print("before savefig")
        figm2.tight_layout()
        plt.savefig(dirName_b0 + figname_b0 + "_LOC" + str(ip) +
                    "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[jindx, iindx],
                                                    dmet.latitude[jindx, iindx]) + ".png")

        plt.close()

        print("\n###########################\n"
              "\nDONE meteogram \n"
              "\n###########################\n")

    def meteogram_average(self, indx, dirName_b2, figname_b2, sitename):
        dmet = self.dmet
        figma1, (axma1, axma2, axma3, axma4) = plt.subplots(nrows=4, ncols=1, figsize=(12, 14), sharex=True)

        def autolabel(rects, axis, fmt='{0:.1f}', space=2):  # for the precip. Got from e
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
        # temp
        temp2m_mean = np.mean(dmet.air_temperature_2m[:, 0, indx[0], indx[1]], axis=(1))

        T2M_MEAN = axma1.plot(dmet.time_normal, temp2m_mean - 273.15, color="red", linewidth=3)
        T2M = axma1.plot(dmet.time_normal, dmet.air_temperature_2m[:, 0, indx[0], indx[1]] - 273.15, color="red",
                         linewidth=0.2,
                         alpha=0.7)

        axma1.set_ylabel('Temp ($^\circ$C)', color="red")
        axma1.tick_params(axis="y", colors="red")
        axma1.set_ylim(bottom=-25, top=0)
        min = np.min([dmet.air_temperature_2m[:, 0, indx[0], indx[1]] - 273.15])
        max = np.max([dmet.air_temperature_2m[:, 0, indx[0], indx[1]] - 273.15])
        axma1.set_ylim(bottom=np.floor(min), top=np.ceil(max))
        # rh
        axma1_2 = axma1.twinx()
        relhum2m_mean = np.mean(dmet.relative_humidity_2m[:, 0, indx[0], indx[1]], axis=(1))
        RH2m_MEAN = axma1_2.plot(dmet.time_normal, relhum2m_mean * 100, color="seagreen", linewidth=3)
        RH2M = axma1_2.plot(dmet.time_normal, dmet.relative_humidity_2m[:, 0, indx[0], indx[1]] * 100, color="seagreen",
                            linewidth=0.2, alpha=0.7)
        axma1_2.set_ylabel(' 2m Rel. Hum. (%)', color="seagreen")
        axma1_2.set_ylim(bottom=0.001, top=100)
        axma1_2.tick_params(axis="y", colors="seagreen")
        # precip
        axma1_3 = axma1.twinx()
        wd = -0.125
        precip_mean = np.mean(dmet.precip1h[:, 0, indx[0], indx[1]], axis=(1))
        precip_max = np.max(dmet.precip1h[:, 0, indx[0], indx[1]], axis=(1))
        precip_min = np.min(dmet.precip1h[:, 0, indx[0], indx[1]], axis=(1))
        P_bar_max = axma1_3.bar(dmet.time_normal, precip_max, color="lightblue", alpha=0.5, width=wd / 4, align="edge",
                                bottom=0,
                                zorder=10)
        P_bar = axma1_3.bar(dmet.time_normal, precip_mean, color="blue", alpha=0.5, width=wd / 4, align="edge",
                            bottom=0,
                            zorder=11)
        # P_bar_min = axma1_3.bar(time_normal, precip_min, color="red", alpha=1, width=wd / 4, align="edge", bottom = 0, zorder=12)
        autolabel(P_bar_max, axma1_3, space=1)
        # autolabel(P_bar, axma1_3, space=1)
        topidx = 1
        maxp = np.nanmax(precip_max[:])
        if maxp > topidx:
            topidx = round_up(maxp)
        axma1_3.set_ylim(bottom=0.001, top=topidx + 0.2 * topidx)

        axma1_3.set_yticks([])
        axma1_3.legend([T2M_MEAN[0], RH2m_MEAN[0], P_bar[0]],
                       ["2m mean Temp.", "2m mean RH", "1h acc mean/max precip"], loc='upper left').set_zorder(99999)
        # xfmt = mdates.DateFormatter('%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        # axma1.xaxis.set_major_formatter(xfmt)
        #################################
        # P2: Spec Hum. and cloudtype
        #################################
        q_mean2m = np.mean(dmet.specific_humidity_2m[:, 0, indx[0], indx[1]], axis=(1))
        Q2M_mean = axma2.plot(dmet.time_normal, q_mean2m * 1000,
                              zorder=0, color="green", alpha=1, linewidth=3)
        Q2M = axma2.plot(dmet.time_normal, dmet.specific_humidity_2m[:, 0, indx[0], indx[1]] * 1000,
                         zorder=0, color="green", alpha=0.7, linewidth=0.2)
        axma2.set_ylabel('Spec. Hum. (g/kg)', color="green")
        axma2.tick_params(axis="y", colors="green")
        maxq = np.max(dmet.specific_humidity_ml[:, -1, indx[0], indx[1]] * 1000)
        topidx = 1.5
        if maxq > 1.5:
            topidx = round_up(maxq, 1)
        axma2.set_ylim(bottom=0, top=topidx)

        axma2_1 = axma2.twinx()
        # stem
        all_low_clf = np.mean(dmet.low_type_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))
        all_mid_clf = np.mean(dmet.medium_type_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))
        all_high_clf = np.mean(dmet.high_type_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))
        all_conv_clf = np.mean(dmet.convective_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))
        tot_all = np.mean(dmet.cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))

        tot_clf_f = axma2_1.fill_between(dmet.time_normal, 0, tot_all * 100, zorder=1, color="gray", alpha=0.6)
        tot_clf = axma2_1.plot(dmet.time_normal, tot_all * 100, zorder=1, color="k")
        tot_patch = mpl.patches.Patch(color='gray', alpha=0.6, linewidth=0)

        conv_clf = axma2_1.plot(dmet.time_normal, all_conv_clf * 100, zorder=2, color="pink")
        high_clf = axma2_1.plot(dmet.time_normal, all_high_clf * 100, zorder=2, color="lightblue", marker=r"$C_H$",
                                markersize=12)
        med_clf = axma2_1.plot(dmet.time_normal, all_mid_clf * 100, zorder=2, color="blue", marker=r"$C_M$",
                               markersize=12)
        low_clf = axma2_1.plot(dmet.time_normal, all_low_clf * 100, zorder=2, color="red", marker=r"$C_L$",
                               markersize=12)
        axma2_1.set_ylabel('Cloud cover %', color="k")
        axma2_1.tick_params(axis="y", colors="k")
        # label
        axma2_1.legend([Q2M[0], (tot_clf[0], tot_patch), high_clf[0], med_clf[0], low_clf[0], conv_clf[0]],
                       ["2m Spec.Hum.", "tot.cloud", "hi. cloud", "med. cloud", "low cloud", "conv. cloud"],
                       loc='upper left').set_zorder(99999)

        ## label
        # axma2.legend([Q2M_mean[0]], ["2m mean Spec.Hum"],loc='upper left').set_zorder(99999)

        #################################
        # P3: Wind ans pressure
        #################################
        wspeed_gust = np.sqrt(
            dmet.x_wind_gust_10m[:, 0, indx[0], indx[1]] ** 2 + dmet.y_wind_gust_10m[:, 0, indx[0], indx[1]] ** 2)
        wspeed = np.sqrt(dmet.x_wind_10m[:, 0, indx[0], indx[1]] ** 2 + dmet.y_wind_10m[:, 0, indx[0], indx[1]] ** 2)
        wsg_mean = np.mean(wspeed_gust, axis=(1))
        ws_mean = np.mean(wspeed, axis=(1))
        WIND_MEAN = axma3.plot(dmet.time_normal, ws_mean, zorder=1, color="darkmagenta", linewidth=3, alpha=1)
        WIND = axma3.plot(dmet.time_normal, wspeed, zorder=1, color="darkmagenta", linewidth=0.2, alpha=0.7)
        GUST_MEAN = axma3.plot(dmet.time_normal, wsg_mean, zorder=0, color="magenta", linewidth=3, alpha=1)
        GUST = axma3.plot(dmet.time_normal, wspeed_gust, zorder=0, color="magenta", linewidth=0.2, alpha=0.7)
        axma3.set_ylabel('wind (m/s)')
        axma3.set_ylim(bottom=0, top=25)
        axma3.tick_params(axis="y", color="darkmagenta")

        axma3_3 = axma3.twinx()
        p_mean = np.mean(dmet.surface_air_pressure[:, 0, indx[0], indx[1]], axis=(1))
        PP = axma3_3.plot(dmet.time_normal, p_mean / 100, zorder=1, color="k", linewidth=3)
        axma3_3.set_ylabel(' Surface Pressure (hPa)')

        axma3_3.legend([GUST_MEAN[0], WIND_MEAN[0], PP[0]],
                       ["10m wind gust", "10m wind (10min mean)", "mean surf. pressure"], loc='upper left').set_zorder(
            99999)

        #################################
        # P4: FLux of sensible/latent heat
        #################################
        SH_mean = np.mean(dmet.H[:, indx[0], indx[1]], axis=(1))
        LH_mean = np.mean(dmet.LE[:, indx[0], indx[1]], axis=(1))
        P_SH_MEAN = axma4.plot(dmet.time_normal, SH_mean, zorder=0, color="blue", linewidth=3, alpha=1)
        P_SH = axma4.plot(dmet.time_normal, dmet.H[:, indx[0], indx[1]], zorder=0, color="blue", linewidth=0.2,
                          alpha=0.7)
        P_LH_MEAN = axma4.plot(dmet.time_normal, LH_mean, zorder=0, color="orange", linewidth=3, alpha=1)
        P_LH = axma4.plot(dmet.time_normal, dmet.LE[:, indx[0], indx[1]], zorder=0, color="orange", linewidth=0.2,
                          alpha=0.7)
        axma4.set_ylabel('Fluxes (W/m$^2$)')
        axma4.legend([P_SH_MEAN[0], P_LH_MEAN[0]], ["Sensible Heat Flux", "Latent Heat Flux"],
                     loc='upper left').set_zorder(
            99999)
        # axma4.xaxis.set_major_formatter(xfmt)

        axma4_1 = axma4.twinx()

        tot_rain = np.nansum(dmet.rainfall_amount[:, -1, indx[0], indx[1]], axis=(1))
        tot_snow = np.nansum(dmet.snowfall_amount[:, -1, indx[0], indx[1]], axis=(1))
        tot_graupel = np.nansum(dmet.graupelfall_amount[:, -1, indx[0], indx[1]], axis=(1))

        tot = tot_rain + tot_snow + tot_graupel
        rain_frac = (tot_rain / tot) * 100
        snow_frac = (tot_snow / tot) * 100
        graupel_frac = (tot_graupel / tot) * 100

        rain_in = axma4_1.plot(dmet.time_normal, rain_frac, zorder=1, color="red", linestyle='dashed', marker='o')
        snow_in = axma4_1.plot(dmet.time_normal, snow_frac, zorder=1, color="gray", linestyle='dashed', marker='*')
        graupel_in = axma4_1.plot(dmet.time_normal, graupel_frac, zorder=1, color="lightblue", linestyle='dashed',
                                  marker='D')
        axma4_1.tick_params(axis="y", colors="k")
        axma4_1.set_ylabel('% of instantaneous precip. type')
        # label
        axma4_1.legend([P_SH[0], P_LH[0], rain_in[0], snow_in[0], graupel_in[0]],
                       ["Sensible H.Flux", "Latent H.Flux", "Rain", "Snow", "Graupel"],
                       loc='upper left').set_zorder(99999)

        #################################
        # SET ADJUSTMENTS ON AXIS
        #################################
        xfmt = mdates.DateFormatter(
            '%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_maj = mdates.DateFormatter('%d.%m')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_min = mdates.DateFormatter('%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

        axma4.xaxis.set_major_locator(mdates.DayLocator())
        axma4.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axma4.xaxis.set_major_formatter(xfmt_maj)
        axma4.xaxis.set_minor_formatter(xfmt_min)

        axma1.xaxis.grid(True, which="major", linewidth=2)
        axma1.xaxis.grid(True, which="minor", linestyle="--")
        axma2.xaxis.grid(True, which="major", linewidth=2)
        axma2.xaxis.grid(True, which="minor", linestyle="--")
        axma3.xaxis.grid(True, which="major", linewidth=2)
        axma3.xaxis.grid(True, which="minor", linestyle="--")
        axma4.xaxis.grid(True, which="major", linewidth=2)
        axma4.xaxis.grid(True, which="minor", linestyle="--")

        axma4.tick_params(axis="x", which="major", pad=12)

        figma1.tight_layout()
        plt.savefig(dirName_b2 + figname_b2 + "_LOC[" + sitename + "]" + ".png")
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
    parser.add_argument("--steps", default=[0, 10], nargs="+", type=int,
                        help="forecast times example --steps 0 3 gives time 0 to 3")
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument("--domain_name", default=None, help="see domain.py")
    parser.add_argument("--domain_lonlat", default=None, nargs="+", type=float, help="lonmin lonmax latmin latmax")
    parser.add_argument("--point_name", default=None, help="see sites.csv")
    parser.add_argument("--point_lonlat", default=None, nargs="+", type=float, help="lon lat")
    parser.add_argument("--point_num", default=1, type=int)
    parser.add_argument("--plot", default="all", help="Display legend")
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()

    for dt in args.datetime:
        dirName_b0, dirName_b1, dirName_b2, dirName_b3, figname_b0, figname_b1, figname_b2, figname_b3 = setup_directory(
            dt, args.point_name, args.point_lonlat)

        VM = PMET(date=dt, steps=args.steps, model=args.model, domain_name=args.domain_name,
                      domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, num_point=args.point_num,
                      point_lonlat=args.point_lonlat, point_name=args.point_name)
        VM.retrieve_handler()
        VM.calculations()
        points, indx_sea, indx_land, indx_alldomain, index_neares = VM.points()
        ip = 0
        for po in points:
            jindx, iindx = po
            VM.plot_meteogram(jindx, iindx, dirName_b0, figname_b0, ip)
            ip += 1

        averagesite = ["ALL_DOMAIN", "ALL_NEAREST", "LAND", "SEA"]  # "ALL_NEAREST", "LAND", "SEA",

        VM.meteogram_average( indx_sea, dirName_b3, figname_b3,  "SEA")
        #plot_maplocation(dmet, data_domain, indx_sea, dirName_b2, figname_b2, sitename, point_lonlat, all=True)
        VM.meteogram_average(indx_land, dirName_b3, figname_b3, "LAND")
        #plot_maplocation(dmet, data_domain, indx_land, dirName_b2, figname_b2, sitename, point_lonlat, all=True)
        VM.meteogram_average(index_neares, dirName_b3, figname_b3, "ALL_NEAREST")
        #plot_maplocation(dmet, data_domain, [jindx, iindx], dirName_b2, figname_b2, sitename, point_lonlat,all=True)
        VM.meteogram_average(indx_alldomain, dirName_b3, figname_b3, "ALL_DOMAIN")
        #plot_maplocation(dmet, data_domain, indx_alldomain, dirName_b2, figname_b2, sitename, point_lonlat,all=True)
