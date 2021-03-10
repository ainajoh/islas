#python point_meteogram.py --datetime 2021022800 --point_name Tromso --domain_name Tromso --steps 0 66

from weathervis.config import *
from weathervis.utils import *
#from weathervis.domain import *
#from weathervis.get_data import *
#from weathervis.calculation import *
from weathervis.checkget_data_handler import *
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import pandas as pd
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#from cartopy.io import shapereader  # For reading shapefiles containg high-resolution coastline.
import warnings

# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)

print(matplotlib.__version__)
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
def setup_met_directory(modelrun, point_name, point_lonlat):
    projectpath = setup_directory(OUTPUTPATH, "{0}/".format(modelrun))
    print("OROOOOJJ")
    print(projectpath)
    figname = "fc_" + modelrun
    # dirName = projectpath + "result/" + modelrun[0].strftime('%Y/%m/%d/%H/')
    if point_lonlat:
        #dirName = projectpath + "/meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_lonlat)
        dirName = projectpath #+ "/"+ str(point_lonlat)

    else:
        #dirName = projectpath + "/meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_name)
        dirName = projectpath #+ "/"+ str(point_name)


    dirName_b1 = dirName #+ "met/"
    figname_b1 = "vmet_" + figname

    dirName_b0 = dirName# + "met/"
    figname_b0 = "met_" + figname

    dirName_b2 = dirName #+ "map/"
    figname_b2 = "map_" + figname

    dirName_b3 = dirName #+ "met/"
    figname_b3 = "met_" + figname

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
        self.param_sfx = ["H", "LE", "SST"]
        self.param = self.param_ml + self.param_pl + self.param_sfc + self.param_sfx
        self.p_level = None
        self.m_level = [64]
        self.mbrs = None
        self.url = None
        self.point_lonlat = point_lonlat
        self.num_point = num_point
        date = str(date)

    def retrieve_handler(self):
        dmet_sfx = None;
        dmet_ml = None;
        dmet_pl = None;
        dmet_sfc = None;
        dmet = None
        split = False
        print("\n######## Checking if your request is possible ############")
        self.param = self.param_pl + self.param_ml + self.param_sfc + self.param_sfx
        dmet,data_domain,bad_param = checkget_data_handler(all_param=self.param, date=self.date, model=self.model, step=self.steps,
                                     p_level=self.p_level, m_level=self.m_level,mbrs=self.mbrs,
                                     domain_name=self.domain_name, domain_lonlat=self.domain_lonlat)

        self.dmet = dmet
        self.data_domain = data_domain

    def calculations(self):
        self.dmet.time_normal = timestamp2utc(self.dmet.time)
        self.dmet.precip1h = precip_acc(self.dmet.precipitation_amount_acc, acc=1)
        self.dmet.ug_10m, self.dmet.vg_10m = xwind2uwind(self.dmet.x_wind_gust_10m, self.dmet.y_wind_gust_10m, self.dmet.alpha)
        self.dmet.u_10m, self.dmet.v_10m = xwind2uwind(self.dmet.x_wind_10m, self.dmet.y_wind_10m, self.dmet.alpha)
        #FOR INFIELD CAMPAIGN
        self.dmet.Tv=None
        self.dmet.rho=None
        self.dmet.samplesize=None
        self.dmet.Tv = virtual_temp(self.dmet.air_temperature_2m, self.dmet.specific_humidity_2m)
        self.dmet.rho = density(self.dmet.Tv, self.dmet.surface_air_pressure)
        self.dmet.samplesize = get_samplesize(self.dmet.specific_humidity_2m, self.dmet.rho, a=0.5, b = 0.95, acc = 3)
        #dexcess
        self.dmet.dexcess=None
        self.dmet.dexcess=dexcess(self.dmet.surface_air_pressure,self.dmet.SST, self.dmet.specific_humidity_2m)
        self.dmet.land_area_fraction =self.dmet.land_area_fraction.squeeze()
        self.dmet.SST =self.dmet.SST.squeeze()



    def points(self):
        point_lonlat = self.point_lonlat; dmet = self.dmet; num_point = self.num_point
        if point_lonlat:
            ind_list = nearest_neighbour(point_lonlat[0], point_lonlat[1], dmet.longitude, dmet.latitude, num_point)
        else:
            sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
            print(sites)
            ind_list = nearest_neighbour(sites.loc[self.point_name].lon, sites.loc[self.point_name].lat, dmet.longitude,
                                         dmet.latitude, num_point)
            self.point_lonlat = [sites.loc[self.point_name].lon, sites.loc[self.point_name].lat]
        poi = ind_list[0:self.num_point]
        #SST = dmet.SST.filled(np.nan)
        #indx_sea = np.where(dmet.SST[0,:,:].mask)
        #indx_land = np.where(~dmet.SST[0,:,:].mask)
        #print("indx_land")
        #
        #print(indx_land)

        indx_sea = np.where( (self.dmet.land_area_fraction[0,:,:] == 0) & (~self.dmet.SST[0,:,:].mask) )
        indx_land = np.where( (self.dmet.land_area_fraction[0,:,:] == 1) & (self.dmet.SST[0,:,:].mask) )
        indx_alldomain = np.where(dmet.latitude != None)
        ll = np.array([list(item) for item in ind_list[0:num_point]])
        jindx = ll[:, 0]
        iindx = ll[:, 1]
        index_neares = [jindx, iindx]
        return poi, indx_sea, indx_land, indx_alldomain, index_neares

    def plot_meteogram(self,jindx, iindx, dirName_b0, figname_b0, ip, indx_sea, indx_land,ndx_alldomain, index_neares):
        dmet=self.dmet
        print("\n###########################\n"
              "\nINITIALISING PLOTTING: meteogram \n"
              "\n###########################\n")
        figm2, (axm1, axm2, axm3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
        figm3, (axm4, axm5, axm6) = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)


        def autolabel(rects, axis, fmt='{0:.1f}', space=2):  # for the precip. Got from e
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects[::space]:
                height = rect.get_height()
                axis.annotate(fmt.format(height).strip('-').strip('0'),
                              xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom')

        #PLOT1##################################################################################################
        print("#PLOT1##################################################################################################")
        subplot1_labels = []
        labeltext = []
        def Temp(air_temperature_2m=True, air_temperature_ml=True, air_temperature_0m = True, subplot1_labels=[],
                 labeltext=[]):
            print("temp plot")
            if dmet.air_temperature_2m is not None:
                T2M = axm1.plot(dmet.time_normal, dmet.air_temperature_2m[:, -1, jindx, iindx] - 273.15, color="brown")
                min = np.min([dmet.air_temperature_2m[:, -1, jindx, iindx]]) - 273.15
                max = np.max([dmet.air_temperature_2m[:, -1, jindx, iindx]]) - 273.15
                subplot1_labels += [T2M[0]]
                labeltext += ["2m Temp."]

            if dmet.air_temperature_ml is not None:
                TML0 = axm1.plot(dmet.time_normal, dmet.air_temperature_ml[:, -1, jindx, iindx] - 273.15, "--", color="brown")
                axm1.set_ylabel('Temp ($^\circ$C)', color="brown")
                axm1.tick_params(axis="y", colors="brown")
                min = np.min([dmet.air_temperature_ml[:, -1, jindx, iindx]]) - 273.15
                max = np.max([dmet.air_temperature_ml[:, -1, jindx, iindx]]) - 273.15
                subplot1_labels += [TML0[0]]
                labeltext += ["10m Temp."]

            if dmet.air_temperature_0m is not None:
                TS = axm1.plot(dmet.time_normal, dmet.air_temperature_0m[:, -1, jindx, iindx] - 273.15, "-.", color="brown")
                axm1.set_ylabel('Temp ($^\circ$C)', color="brown")
                axm1.tick_params(axis="y", colors="brown")
                min = np.min([dmet.air_temperature_ml[:, -1, jindx, iindx], dmet.air_temperature_0m[:, -1, jindx, iindx]]) - 273.15
                max = np.max([dmet.air_temperature_ml[:, -1, jindx, iindx], dmet.air_temperature_0m[:, -1, jindx, iindx]]) - 273.15
                subplot1_labels += [TS[0]]
                labeltext += ["0m Temp"]

            if dmet.air_temperature_2m is None and dmet.air_temperature_ml is None and dmet.air_temperature_0m is None:
                print("NO temperature plot available")
            else:
                axm1.set_ylim(bottom=np.floor(min), top=np.ceil(max))
            return axm1, subplot1_labels,labeltext
        def RH():
            print("in rh")
            if dmet.relative_humidity_2m is not None:
                axm1_2 = axm1.twinx()
                axm1_2.plot(dmet.time_normal, dmet.relative_humidity_2m[:, 0, jindx, iindx] * 100, color="seagreen", alpha=0.8)
                axm1_2.set_ylabel(' 2m Rel. Hum. (%)', color="seagreen")
                axm1_2.set_ylim(bottom=0.001, top=100)
                axm1_2.tick_params(axis="y", colors="seagreen")
            else:
                print("No rh 2m ")
            return axm1_2, subplot1_labels, labeltext
        def Precip(subplot1_labels=[], labeltext=[]):
            print("in precip")
            if dmet.precip1h is not None:
                axm1_3 = axm1.twinx()
                wd = -0.125
                P_bar = axm1_3.bar(dmet.time_normal, dmet.precip1h[:, 0, jindx, iindx], color="blue", alpha=0.8, width=wd / 4,
                                   align="edge")
                topidx = 1.
                maxp = np.nanmax(dmet.precip1h[:, 0, jindx, iindx])
                if maxp > topidx:
                    topidx = round_up(maxp)
                axm1_3.set_ylim(bottom=0.001,
                            top=topidx + 0.1 * topidx)  # adding 20% of max value for getting some clearense above bar
                autolabel(P_bar, axm1_3, space=2)
                axm1_3.set_yticks([])
                axm1_3.set_xticks([])
                subplot1_labels += [P_bar[0]]
                labeltext += ["1h acc precip"]
                return axm1_3, subplot1_labels, labeltext
        axm1, subplot1_labels,labeltext = Temp(subplot1_labels=subplot1_labels,labeltext=labeltext)
        axm1_2, subplot1_labels,labeltext = RH()
        axm1_3, subplot1_labels,labeltext = Precip(subplot1_labels=subplot1_labels,labeltext=labeltext)
        axm1_3.legend(subplot1_labels,labeltext,loc='upper left').set_zorder(99999)
        axm1.xaxis.grid(True, which="major", linewidth=2)
        axm1.xaxis.grid(True, which="minor", linestyle="--")
        #PLOT2##################################################################################################
        print("#PLOT2###########################################################################################")
        subplot2_labels = []
        labeltext2 = []
        def q(subplot2_labels=[],labeltext2=[]):
            print("in q")
            if dmet.specific_humidity_2m is not None:
                Q2M = axm2.plot(dmet.time_normal, dmet.specific_humidity_2m[:, 0, jindx, iindx] * 1000, zorder=2, color="green",
                             alpha=0.8)
                maxq = np.max(dmet.specific_humidity_2m[:, 0, jindx, iindx] * 1000)
                subplot2_labels += [Q2M[0]]
                labeltext2 += ["2m Spec.Hum."]

            if dmet.specific_humidity_ml is not None:
                QML0 = axm2.plot(dmet.time_normal, dmet.specific_humidity_ml[:, -1, jindx, iindx] * 1000, "--", zorder=2,
                                 color="green",
                                 alpha=0.8)
                maxq = np.max(dmet.specific_humidity_ml[:, -1, jindx, iindx] * 1000)
                subplot2_labels += [QML0[0]]
                labeltext2 += ["10m Spec.Hum."]


            if dmet.specific_humidity_2m is None and dmet.specific_humidity_ml is None:
                print("no spec.hum avail.")
            else:
                axm2.set_ylabel('Spec. Hum. (g/kg)', color="green")
                axm2.tick_params(axis="y", colors="green")
                topidx = 1.5
                if maxq > 1.5:
                    topidx = round_up(maxq, 1)
                axm2.set_ylim(bottom=0, top=topidx + 0.10 * topidx)
            return axm2, subplot2_labels, labeltext2
        def CloudType(subplot2_labels=[],labeltext2=[]):
            print("in cloud")
            axm2_1 = axm2.twinx()
            if dmet.cloud_area_fraction is not None:
                tot_clf_f = axm2_1.fill_between(dmet.time_normal, 0, dmet.cloud_area_fraction[:, -1, jindx, iindx] * 100,
                                        zorder=1, color="gray", alpha=0.6)
                tot_clf = axm2_1.plot(dmet.time_normal, dmet.cloud_area_fraction[:, -1, jindx, iindx] * 100, zorder=1,
                              color="k")
                tot_patch = mpl.patches.Patch(color='gray', alpha=0.5, linewidth=0)
                subplot2_labels += [(tot_clf[0], tot_patch)]
                labeltext2 += ["tot.cloud"]

            if dmet.high_type_cloud_area_fraction is not None:
                #conv_clf = axm2_1.plot(dmet.time_normal, dmet.convective_cloud_area_fraction[:, -1, jindx, iindx] * 100,
                #               zorder=2, color="pink")
                high_clf = axm2_1.plot(dmet.time_normal, dmet.high_type_cloud_area_fraction[:, -1, jindx, iindx] * 100,
                               zorder=2, color="lightblue", marker=r"$C_H$", markersize=12)
                med_clf = axm2_1.plot(dmet.time_normal, dmet.medium_type_cloud_area_fraction[:, -1, jindx, iindx] * 100,
                              zorder=2, color="blue", marker=r"$C_M$", markersize=12)
                low_clf = axm2_1.plot(dmet.time_normal, dmet.low_type_cloud_area_fraction[:, -1, jindx, iindx] * 100, zorder=2,
                              color="red", marker=r"$C_L$", markersize=12)
                subplot2_labels += [high_clf[0], med_clf[0], low_clf[0]] #, conv_clf[0]
                labeltext2 += ["hi. cloud", "med. cloud", "low cloud"] #, "conv. cloud"
            if dmet.convective_cloud_area_fraction is None and dmet.cloud_area_fraction is None:
                print("NO cloudType aval.")
            else:
                axm2_1.set_ylabel('Cloud cover %', color="k")
                axm2_1.tick_params(axis="y", colors="k")
                # label
            return axm2_1, subplot2_labels, labeltext2
        axm2, subplot2_labels, labeltext2 = q(subplot2_labels=subplot2_labels, labeltext2=labeltext2)
        axm2_1, subplot2_labels, labeltext2 = CloudType(subplot2_labels=subplot2_labels, labeltext2=labeltext2)
        axm2_1.legend(subplot2_labels,labeltext2,loc='upper left').set_zorder(99999)
        #PLOT3##################################################################################################
        print("#PLOT3###########################################################################################")
        subplot3_labels = []
        labeltext3 = []
        def wind(subplot3_labels = [], labeltext3 = []):
            if dmet.u_10m is not None:
                wspeed = np.sqrt(dmet.u_10m[:, 0, jindx, iindx] ** 2 + dmet.v_10m[:, 0, jindx, iindx] ** 2)
                WIND = axm3.plot(dmet.time_normal, wspeed, zorder=1, color="darkmagenta")
                axm3.quiver(dmet.time_normal, wspeed, dmet.u_10m[:, 0, jindx, iindx] / wspeed,
                            dmet.v_10m[:, 0, jindx, iindx] / wspeed, scale=80, zorder=2)
                subplot3_labels += [WIND[0]]
                labeltext3 += ["10m wind (10min mean)"]
            if dmet.ug_10m is not None:
                wspeed_gust = np.sqrt(dmet.ug_10m[:, 0, jindx, iindx] ** 2 + dmet.vg_10m[:, 0, jindx, iindx] ** 2)
                GUST = axm3.plot(dmet.time_normal, wspeed_gust, zorder=1, color="magenta")
                axm3.quiver(dmet.time_normal, wspeed_gust, dmet.ug_10m[:, 0, jindx, iindx] / wspeed_gust,
                        dmet.vg_10m[:, 0, jindx, iindx] / wspeed_gust, scale=80, zorder=2)
                subplot3_labels += [GUST[0]]
                labeltext3 += ["10m wind gust"]
            if dmet.u_10m is None and dmet.ug_10m is None:
                print("NO cloudType aval.")
            else:
                axm3.set_ylabel('wind (m/s)')
                axm3.set_ylim(bottom=0, top=25)
                axm3.tick_params(axis="y", color="darkmagenta")
            return axm3, subplot3_labels, labeltext3
        def pressure(subplot3_labels = [], labeltext3 = []):
            # pressure
            if pressure is not None:
                axm3_2 = axm3.twinx()
                P = axm3_2.plot(dmet.time_normal, dmet.surface_air_pressure[:, 0, jindx, iindx] / 100, zorder=1, color="k")
                axm3_2.set_ylabel(' Surface Pressure (hPa)')
            subplot3_labels += [P[0]]
            labeltext3 += ["mean. surf. pressure"]
            return axm3_2, subplot3_labels, labeltext3
        axm3, subplot3_labels, labeltext3 = wind(subplot3_labels, labeltext3)
        axm3_2, subplot3_labels, labeltext3 = pressure(subplot3_labels, labeltext3)
        axm3_2.legend(subplot3_labels, labeltext3, loc='upper left').set_zorder(99999)
        #FIG2 PLOT4###############################################################################################
        print("#FIG2 PLOT4########################################################################################")
        subplot4_labels = []
        labeltext4 = []
        def fluxes(subplot4_labels = [], labeltext4 = []):
            if dmet.H is not None:
                P_SH = axm4.plot(dmet.time_normal, dmet.H[:, jindx, iindx], zorder=1, color="blue")
                P_LH = axm4.plot(dmet.time_normal, dmet.LE[:, jindx, iindx], zorder=1, color="orange")
                axm4.set_ylabel('Heat Fluxes (W/m$^2$)')
                subplot4_labels += [P_SH[0], P_LH[0]]
                labeltext4 += ["Sensible H.Flux", "Latent H.Flux"]
                # rainfall_amount
            return axm4, subplot4_labels, labeltext4
        def precip_type(subplot4_labels = [], labeltext4 = []):
            axm4_1 = axm4.twinx()
            if dmet.snowfall_amount is not None:
                tot = dmet.rainfall_amount[:, -1, jindx, iindx] + dmet.snowfall_amount[:, -1, jindx,
                                                          iindx] + dmet.graupelfall_amount[:, -1, jindx, iindx]
                rain_frac=\
                    np.divide(dmet.rainfall_amount[:, -1, jindx, iindx],tot, out=np.zeros(np.shape(dmet.rainfall_amount[:, -1, jindx, iindx])), where=tot!=0.)*100
                snow_frac=\
                    np.divide(dmet.snowfall_amount[:, -1, jindx, iindx],tot, out=np.zeros(np.shape(dmet.snowfall_amount[:, -1, jindx, iindx])), where=tot!=0.)*100
                graupel_frac=\
                    np.divide(dmet.graupelfall_amount[:, -1, jindx, iindx],tot, out=np.zeros(np.shape(dmet.graupelfall_amount[:, -1, jindx, iindx])), where=tot!=0.)*100

                wd = -0.125

                p1 = axm4_1.bar(dmet.time_normal, rain_frac,width=wd / 4,alpha=0.3, color="red", zorder=0)
                p2 = axm4_1.bar(dmet.time_normal, snow_frac,width=wd / 4, bottom=rain_frac,alpha=0.3, color="aqua", zorder=0)
                p3 = axm4_1.bar(dmet.time_normal, graupel_frac,width=wd / 4, bottom=rain_frac+snow_frac,alpha=0.3, color="gray", zorder=0)

                subplot4_labels += [p1[0], p2[0], p3[0]]
                labeltext4 += [ "Rain", "Snow", "Graupel"]


            return axm4_1, subplot4_labels, labeltext4
        axm4, subplot4_labels, labeltext4 = fluxes(subplot4_labels, labeltext4)
        axm4_1, subplot4_labels, labeltext4 = precip_type(subplot4_labels, labeltext4)
        axm4_1.tick_params(axis="y", colors="k")
        axm4_1.set_ylabel('% of instantaneous precip. type')
        axm4_1.legend(subplot4_labels, labeltext4, loc='upper left').set_zorder(99999)
        #FIG2 PLOT5###############################################################################################
        print("#PLOT5##################################################################################################")
        subplot5_labels = []
        labeltext5 = []
        def samples(subplot5_labels = [], labeltext5 = []):
            if dmet.samplesize is not None:
                print(np.shape(dmet.samplesize))
                print(np.shape(self.dmet.samplesize))
                wd = -0.125  # width of 3h x axis found emprically, should heve better automatic
                S_bar = axm5.bar(dmet.time_normal, dmet.samplesize[:, jindx, iindx], ls="--",
                                   ec="green", color="lightblue",
                                   alpha=0.8, width=wd/4, align="edge")
                autolabel(S_bar, axis=axm5, fmt='{:.3f}', space=3)
                axm5.set_yticks([])
                #axm5.set_axis_off()
                axm5.set_ylim(bottom=0, top=1.2)
                subplot5_labels += [S_bar[0]]
                labeltext5 += ["Smaple size (g/3h)"]
                # rainfall_amount
            return axm5, subplot5_labels, labeltext5
        def plot_dexcess(subplot5_labels = [], labeltext5 = []):
            axm5_1 = axm5.twinx()

            if dmet.dexcess is not None:
                print("dexcess")
                print(np.shape(dmet.dexcess))
                wd = -0.125  # width of 3h x axis found emprically, should heve better automatic
                #indx_sea = np.where(dmet.land_area_fraction[0][0][:][:] == 0)
                #points_sea = [(x,y) for x,y in zip(indx_sea[0],indx_sea[1])]
                #
                #near_sea = nearest_neighbour_idx(self.point_lonlat[0],self.point_lonlat[1],
                #                             dmet.longitude[indx_sea[0],indx_sea[1]],
                #                             dmet.latitude[indx_sea[0],indx_sea[1]])
                S_dx= axm5_1.plot(dmet.time_normal, dmet.dexcess[:, jindx, iindx],color="blue")
                #axm5_1.set_ylim(bottom=0, top=
                axm5_1.set_ylabel(' d-excess ',color="blue")
                axm5_1.tick_params(axis="y", colors="blue")

                # rainfall_amount
            return axm5_1, subplot5_labels, labeltext5
        def plot_sst(subplot5_labels = [], labeltext5 = []):
            axm5_2 = axm5.twinx()
            if dmet.SST is not None:
                S_dx= axm5_2.plot(dmet.time_normal, dmet.SST[:, jindx, iindx], color="brown")
                axm5_2.set_ylabel(' SST ', color="brown")
                axm5_2.tick_params(axis="y", colors="brown")
                axm5_2.yaxis.set_label_position("left")
                axm5_2.yaxis.tick_left()

            return axm5_2, subplot5_labels, labeltext5




        axm5, subplot5_labels, labeltext5 = samples(subplot5_labels, labeltext5)
        axm5_1, subplot5_labels, labeltext5 = plot_dexcess(subplot5_labels, labeltext5)
        axm5_2, subplot5_labels, labeltext5 = plot_sst(subplot5_labels, labeltext5)

        axm5_1.legend(subplot5_labels, labeltext5, loc='upper left').set_zorder(99999)
        #FIG2 PLOT6###############################################################################################
        print("#PLOT6##################################################################################################")
        subplot6_labels = []
        labeltext6 = []
        axm6.remove()
        def loc2map(subplot6_labels = [], labeltext6 = []):
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            axm6_0 = figm3.add_subplot(3,1,3)
            axm6_0.set_xticks([])
            axm6_0.set_yticks([])
            axm6_0.set_facecolor("lightgray")

            lon0 = dmet.longitude_of_central_meridian_projection_lambert
            lat0 = dmet.latitude_of_projection_origin_projection_lambert
            parallels = dmet.standard_parallel_projection_lambert
            globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
            crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                        globe=globe)
            axm6 = figm3.add_subplot(3,3,8,projection=crs)
            axm6.background_patch.set_facecolor('lightskyblue')  # fill_color='lightskyblue'
            axm6.background_patch.set_edgecolor('red')  # fill_color='lightskyblue'
            axm6.background_patch.set_linewidth(5)  # fill_color='lightskyblue'

            axm6.add_feature(cfeature.GSHHSFeature(scale='high'), facecolor='whitesmoke', edgecolor='k', linewidths=2.,
                           zorder=0)
            #CC = axm6.contour(dmet.longitude, dmet.latitude, dmet.land_area_fraction[0, :, :], alpha=0.6, zorder=1,
            #                levels=[0.9, 1, 1.1],
            #                colors="b", linewidths=5, transform=ccrs.PlateCarree())
            mainpoint = axm6.scatter(self.point_lonlat[0], self.point_lonlat[1], c="blue", s=6 ** 2, transform=ccrs.PlateCarree(),zorder=2)
            allpoint = axm6.scatter(dmet.longitude, dmet.latitude, c="black", s=5 ** 2, transform=ccrs.PlateCarree(),zorder=1)
            evalpoint = axm6.scatter(dmet.longitude[jindx,iindx], dmet.latitude[jindx,iindx], c="red", s=6 ** 2, transform=ccrs.PlateCarree(),zorder=4)

            axm6_1 = figm3.add_subplot(3,3,9,projection=crs)
            axm6_1.background_patch.set_facecolor('lightskyblue')  # fill_color='lightskyblue'
            axm6_1.add_feature(cfeature.GSHHSFeature(scale='low'), facecolor='whitesmoke', edgecolor='k', linewidths=1.,
                             zorder=0)
            axm6_1.scatter(dmet.x[iindx], dmet.y[jindx], facecolors='none', edgecolors='r', s=6 ** 2,zorder=1,marker='s')
            axm6_1.set_extent([0, 25, 50, 85], crs=ccrs.PlateCarree())
            axm6_1.set_zorder(4)
            from matplotlib.patches import ConnectionPatch
            con = ConnectionPatch(xyA=( dmet.x[iindx], dmet.y[jindx] ), xyB= (1,1), coordsA="data", coordsB="axes fraction", axesA=axm6_1, axesB=axm6)
            con2 = ConnectionPatch(xyA=( dmet.x[iindx], dmet.y[jindx] ), xyB= (1,0), coordsA="data", coordsB="axes fraction", axesA=axm6_1, axesB=axm6)
            con.set_color("red")
            axm6_1.add_artist(con)  #axm6_1.transData
            con.set_linewidth(1)
            con2.set_color("red")
            axm6_1.add_artist(con2)  # axm6_1.transData
            con2.set_linewidth(1)
            subplot6_labels += [mainpoint, allpoint, evalpoint]
            labeltext6 += [f"requested pos.\n[{self.point_lonlat[0]},{self.point_lonlat[1]} ]\n {self.point_name}", "model gridpoints", "evaluated gridpoint"]
            axm6_L = figm3.add_subplot(3, 3, 7)
            axm6_L.set_xticks([] )
            axm6_L.set_yticks([])
            axm6_L.axis("off")
            #axm6_0.set_zorder(0)
            #axm6.set_zorder(1)
            #axm6_1.set_zorder(2)
            axm6_L.set_zorder(99999)
            #axm6_0.set_facecolor(None)


            return axm6,axm6_1,axm6_0,axm6_L,subplot6_labels, labeltext6
        axm6,axm6_1,axm6_0,axm6_L, subplot6_labels, labeltext6 = loc2map(subplot6_labels, labeltext6)
        axm6_L.legend(subplot6_labels, labeltext6, loc='best').set_zorder(99999)

        #################################
        # SET ADJUSTMENTS ON AXIS
        #################################
        xfmt = mdates.DateFormatter(
            '%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_maj = mdates.DateFormatter('%d.%m')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_min = mdates.DateFormatter('%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

        axm3.xaxis.set_major_locator(mdates.DayLocator())
        axm3.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axm3.xaxis.set_major_formatter(xfmt_maj)
        axm3.xaxis.set_minor_formatter(xfmt_min)
        axm5.xaxis.set_major_locator(mdates.DayLocator())
        axm5.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axm5.xaxis.set_major_formatter(xfmt_maj)
        axm5.xaxis.set_minor_formatter(xfmt_min)
        axm1.xaxis.set_major_locator(mdates.DayLocator())
        axm1.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axm1.xaxis.set_major_formatter(xfmt_maj)
        axm1.xaxis.set_minor_formatter(xfmt_min)
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
        axm5.xaxis.grid(True, which="major", linewidth=2)
        axm5.xaxis.grid(True, which="minor", linestyle="--")

        axm3.tick_params(axis="x", which="major", pad=12, bottom=True,labelbottom=True)
        axm5.tick_params(axis="x", which="major", pad=12, bottom=True,labelbottom=True)
        axm1.tick_params(axis="x", which="major", pad=12,top=True, labeltop=True)
        axm4.tick_params(axis="x", which="major", pad=12,top=True, labeltop=True)


        print("before savefig")
        #figm2.tight_layout()
        loc_name=self.point_name if self.point_name else self.point_lonlat

        print(dirName_b0 + figname_b0 + "_LOC_" + str(loc_name) +
                    "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[jindx, iindx],
                                                    dmet.latitude[jindx, iindx]) + ".png")
        figm2.savefig(dirName_b0 + figname_b0 + "_o1_LOC_" + str(loc_name) +
                    "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[jindx, iindx],
                                                    dmet.latitude[jindx, iindx]) + ".png", bbox_inches = "tight", dpi = 200)

        figm3.savefig(dirName_b0 + figname_b0 + "_o2_LOC_" + str(loc_name) +
                      "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[jindx, iindx],
                                                      dmet.latitude[jindx, iindx]) + ".png", bbox_inches="tight",
                      dpi=200)
        print("fig saved")
        plt.close()

        print("\n###########################\n"
              "\nDONE meteogram \n"
              "\n###########################\n")

    def meteogram_average(self, indx, dirName_b2, figname_b2, sitename):
        dmet = self.dmet
        #figma1, (axma1, axma2, axma3, axma4) = plt.subplots(nrows=4, ncols=1, figsize=(12, 14), sharex=True)
        figma1, (axma1, axma2, axma3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
        figma2, (axma4, axma5, axma6) = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
        def autolabel(rects, axis, fmt='{0:.1f}', space=2):  # for the precip. Got from e
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects[::space]:
                height = rect.get_height()
                axis.annotate(fmt.format(height).strip('-').strip('0'),
                              xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom', zorder=99999)

        #PLOT1##################################################################################################
        print("#PLOT1############################################################################################")
        subplot1_labels = []
        labeltext = []
        def Temp(air_temperature_2m=True, air_temperature_ml=True, air_temperature_0m=True, subplot1_labels=[],
                 labeltext=[]):
            print("TEMP AVG")
            if dmet.air_temperature_2m is not None:
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
                subplot1_labels += [T2M_MEAN[0]]
                labeltext += ["2m mean Temp."]
            else:
                print("No temp available")
            return axma1, subplot1_labels, labeltext


            # rh
        def RH(subplot1_labels=[],labeltext=[]):
            print("RH AVG")

            if dmet.relative_humidity_2m is not None:
                axma1_2 = axma1.twinx()
                relhum2m_mean = np.mean(dmet.relative_humidity_2m[:, 0, indx[0], indx[1]], axis=(1))
                RH2m_MEAN = axma1_2.plot(dmet.time_normal, relhum2m_mean * 100, color="seagreen", linewidth=3)
                RH2M = axma1_2.plot(dmet.time_normal, dmet.relative_humidity_2m[:, 0, indx[0], indx[1]] * 100, color="seagreen",
                               linewidth=0.2, alpha=0.7)
                axma1_2.set_ylabel(' 2m Rel. Hum. (%)', color="seagreen")
                axma1_2.set_ylim(bottom=0.001, top=100)
                axma1_2.tick_params(axis="y", colors="seagreen")
                subplot1_labels += [RH2m_MEAN[0]]
                labeltext += ["2m mean RH."]
            return axma1_2, subplot1_labels, labeltext
        def Precip(subplot1_labels=[], labeltext=[]):
            print("PRECIP AVG")

            axma1_3 = axma1.twinx()
            if dmet.precip1h is not None:
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
                autolabel(P_bar_max, axma1_3, space=2)
                topidx = 1
                maxp = np.nanmax(precip_max[:])
                if maxp > topidx:
                    topidx = round_up(maxp)
                axma1_3.set_ylim(bottom=0.001, top=topidx + 0.2 * topidx)
                axma1_3.set_yticks([])
                subplot1_labels += [P_bar[0]]
                labeltext += ["1h acc mean/max precip"]
            return axma1_3, subplot1_labels, labeltext
        axma1, subplot1_labels, labeltext = Temp(subplot1_labels=subplot1_labels,labeltext=labeltext)
        axma1_2, subplot1_labels, labeltext = RH(subplot1_labels=subplot1_labels,labeltext=labeltext)
        axma1_3, subplot1_labels, labeltext = Precip(subplot1_labels=subplot1_labels,labeltext=labeltext)
        axma1_3.legend(subplot1_labels, labeltext, loc='upper left').set_zorder(99999)
        # PLOT2##################################################################################################
        print("#PLOT2############################################################################################")
        subplot2_labels = []
        labeltext2 = []
        def q(subplot2_labels=[],labeltext2=[]):
            print("Q AVG")

            if dmet.specific_humidity_2m is not None:
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
                subplot2_labels += [Q2M_mean[0]]
                labeltext2 += ["2m Spec.Hum."]
            return axma2,subplot2_labels,labeltext2
        def CloudType(subplot2_labels=[], labeltext2=[]):
            print("CLOUD AVG")

            if dmet.low_type_cloud_area_fraction is not None:
                axma2_1 = axma2.twinx()
                all_low_clf = np.mean(dmet.low_type_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))
                all_mid_clf = np.mean(dmet.medium_type_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))
                all_high_clf = np.mean(dmet.high_type_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))
                all_conv_clf = np.mean(dmet.convective_cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))
                tot_all = np.mean(dmet.cloud_area_fraction[:, -1, indx[0], indx[1]], axis=(1))
                print("tot clf")
                tot_clf_f = axma2_1.fill_between(dmet.time_normal, 0, tot_all * 100, zorder=1, color="gray", alpha=0.6)
                tot_clf = axma2_1.plot(dmet.time_normal, tot_all * 100, zorder=1, color="k")
                tot_patch = mpl.patches.Patch(color='gray', alpha=0.6, linewidth=0)
                print("high clf")

                #conv_clf = axma2_1.plot(dmet.time_normal, all_conv_clf * 100, zorder=2, color="pink")
                high_clf = axma2_1.plot(dmet.time_normal, all_high_clf * 100, zorder=2, color="lightblue", marker=r"$C_H$",
                                markersize=12)
                med_clf = axma2_1.plot(dmet.time_normal, all_mid_clf * 100, zorder=2, color="blue", marker=r"$C_M$",
                               markersize=12)
                low_clf = axma2_1.plot(dmet.time_normal, all_low_clf * 100, zorder=2, color="red", marker=r"$C_L$",
                               markersize=12)
                print("label clf")

                axma2_1.set_ylabel('Cloud cover %', color="k")
                axma2_1.tick_params(axis="y", colors="k")
                subplot2_labels += [(tot_clf[0], tot_patch), high_clf[0], med_clf[0], low_clf[0]] # conv_clf[0]
                labeltext2 += ["tot.cloud", "hi. cloud", "med. cloud", "low cloud"] #, "conv. cloud"
                print("cloud done")
            return axma2_1, subplot2_labels, labeltext2
        axma2, subplot2_labels, labeltext2 = q(subplot2_labels=subplot2_labels, labeltext2=labeltext2)
        axma2_1, subplot2_labels, labeltext2 = CloudType(subplot2_labels=subplot2_labels, labeltext2=labeltext2)
        axma2_1.legend(subplot2_labels,labeltext2,loc='upper left').set_zorder(99999)
        # PLOT3##################################################################################################
        print("#PLOT3############################################################################################")
        subplot3_labels = []
        labeltext3 = []
        def wind(subplot3_labels=[], labeltext3=[]):
            print("wind")
            if dmet.y_wind_10m is not None:
                wspeed = np.sqrt(dmet.x_wind_10m[:, 0, indx[0], indx[1]] ** 2 + dmet.y_wind_10m[:, 0, indx[0], indx[1]] ** 2)
                ws_mean = np.mean(wspeed, axis=(1))
                WIND_MEAN = axma3.plot(dmet.time_normal, ws_mean, zorder=1, color="darkmagenta", linewidth=3, alpha=1)
                WIND = axma3.plot(dmet.time_normal, wspeed, zorder=1, color="darkmagenta", linewidth=0.2, alpha=0.7)
                subplot3_labels += [WIND_MEAN[0]]
                labeltext3 += ["10m wind (10min mean)"]
            if dmet.y_wind_gust_10m is not None:
                wspeed_gust = np.sqrt(dmet.x_wind_gust_10m[:, 0, indx[0], indx[1]] ** 2 + dmet.y_wind_gust_10m[:, 0, indx[0],indx[1]] ** 2)
                wsg_mean = np.mean(wspeed_gust, axis=(1))
                GUST_MEAN = axma3.plot(dmet.time_normal, wsg_mean, zorder=0, color="magenta", linewidth=3, alpha=1)
                GUST = axma3.plot(dmet.time_normal, wspeed_gust, zorder=0, color="magenta", linewidth=0.2, alpha=0.7)
                subplot3_labels += [GUST_MEAN[0]]
                labeltext3 += ["10m wind gust"]

            if dmet.y_wind_10m is None and dmet.y_wind_gust_10m is None:
                print("No wind available")
            else:
                axma3.set_ylabel('wind (m/s)')
                axma3.set_ylim(bottom=0, top=25)
                axma3.tick_params(axis="y", color="darkmagenta")
            return axma3, subplot3_labels,labeltext3
        def pressure(subplot3_labels=[], labeltext3=[]):
            print("pressure")
            axma3_3 = axma3.twinx()
            if dmet.surface_air_pressure is not None:
                p_mean = np.mean(dmet.surface_air_pressure[:, 0, indx[0], indx[1]], axis=(1))
                PP = axma3_3.plot(dmet.time_normal, p_mean / 100, zorder=1, color="k", linewidth=3)
                axma3_3.set_ylabel(' Surface Pressure (hPa)')
                subplot3_labels += [PP[0]]
                labeltext3 += ["mean surf. pressure"]
            return axma3_3,subplot3_labels,labeltext3
        axma3, subplot3_labels, labeltext3 = wind(subplot3_labels=subplot3_labels, labeltext3=labeltext3)
        axma3_3, subplot3_labels, labeltext3 = pressure(subplot3_labels=subplot3_labels, labeltext3=labeltext3)
        axma3_3.legend(subplot3_labels,labeltext3, loc='upper left').set_zorder(99999)
        # FIG2 PLOT4##################################################################################################
        print("#FIG 2 PLOT4##########################################################################################")
        subplot4_labels = []
        labeltext4 = []
        def fluxes(subplot4_labels=[], labeltext4=[]):
            print("fluxes")
            if dmet.H is not None:
                SH_mean = np.mean(dmet.H[:, indx[0], indx[1]], axis=(1))
                LH_mean = np.mean(dmet.LE[:, indx[0], indx[1]], axis=(1))
                P_SH_MEAN = axma4.plot(dmet.time_normal, SH_mean, zorder=1, color="blue", linewidth=3, alpha=1)
                P_SH = axma4.plot(dmet.time_normal, dmet.H[:, indx[0], indx[1]], zorder=1, color="blue", linewidth=0.2,
                          alpha=0.7)
                P_LH_MEAN = axma4.plot(dmet.time_normal, LH_mean, zorder=1, color="orange", linewidth=3, alpha=1)
                P_LH = axma4.plot(dmet.time_normal, dmet.LE[:, indx[0], indx[1]], zorder=1, color="orange", linewidth=0.2,
                          alpha=0.7)
                axma4.set_ylabel('Fluxes (W/m$^2$)')
                #axma4.legend([P_SH_MEAN[0], P_LH_MEAN[0]], ["Sensible Heat Flux", "Latent Heat Flux"],
                #        loc='upper left').set_zorder(99999)
                subplot4_labels = [P_SH_MEAN[0], P_SH_MEAN[0]]
                labeltext4 = ["Sensible H.Flux", "Latent H.Flux"]
            return axma4,subplot4_labels,labeltext4
        def precip_type(subplot4_labels=[], labeltext4=[]):
            print("precio type")
            axma4_1 = axma4.twinx()
            if dmet.snowfall_amount is not None:
                tot_rain = np.nansum(dmet.rainfall_amount[:, -1, indx[0], indx[1]], axis=(1))
                tot_snow = np.nansum(dmet.snowfall_amount[:, -1, indx[0], indx[1]], axis=(1))
                tot_graupel = np.nansum(dmet.graupelfall_amount[:, -1, indx[0], indx[1]], axis=(1))
                print("nansum done")

                tot = tot_rain + tot_snow + tot_graupel

                rain_frac = \
                    np.divide(tot_rain, tot,
                          out=np.zeros(np.shape(tot_rain)), where=tot != 0.) * 100
                snow_frac = \
                    np.divide(tot_snow, tot,
                          out=np.zeros(np.shape(tot_snow)), where=tot != 0.) * 100
                graupel_frac = \
                    np.divide(tot_graupel, tot,
                          out=np.zeros(np.shape(tot_graupel)), where=tot != 0.) * 100

                wd = -0.125
                print("calc done")
                #AVOIDING UserWarning: Warning: converting a masked element to nan.
                #rain_frac = rain_frac.filled(np.nan)
                #snow_frac = snow_frac.filled(np.nan)
                #graupel_frac = graupel_frac.filled(np.nan)

                p1 = axma4_1.bar(dmet.time_normal, rain_frac, width=wd / 4, alpha=0.3, color="red", zorder=0)
                p2 = axma4_1.bar(dmet.time_normal, snow_frac, width=wd / 4, bottom=rain_frac, alpha=0.3, color="aqua",
                                zorder=0)
                p3 = axma4_1.bar(dmet.time_normal, graupel_frac, width=wd / 4, bottom=rain_frac + snow_frac, alpha=0.3,
                                color="gray", zorder=0)
                print("bar done")

                subplot4_labels += [p1[0], p2[0], p3[0]]
                labeltext4 += ["Rain", "Snow", "Graupel"]


                axma4_1.tick_params(axis="y", colors="k")
                axma4_1.set_ylabel('% of instantaneous precip. type')
                #subplot4_labels += [rain_in[0], snow_in[0], graupel_in[0]]
                #labeltext4 += ["Rain", "Snow", "Graupel"]
            print("precio done")
            return axma4_1, subplot4_labels, labeltext4
        axma4, subplot4_labels, labeltext4 = fluxes(subplot4_labels=subplot4_labels, labeltext4=labeltext4)
        axma4_1,subplot4_labels, labeltext4 = precip_type(subplot4_labels=subplot4_labels, labeltext4=labeltext4)
        axma4_1.legend(subplot4_labels,labeltext4, loc='upper left').set_zorder(99999)
        # FIG5 PLOT4##################################################################################################
        print("#FIG 2 PLOT5##########################################################################################")
        subplot5_labels = []
        labeltext5 = []
        def samples(subplot5_labels=[], labeltext5=[]):
            if dmet.samplesize is not None:
                av_tot_sample = np.mean(dmet.samplesize[:, indx[0], indx[1]], axis=(1))
                #av_tot_sample=tot_sample/np.shape(dmet.samplesize)[2]
                print(np.shape(dmet.samplesize))
                print(np.shape(self.dmet.samplesize))
                wd = -0.125  # width of 3h x axis found emprically, should heve better automatic
                S_bar = axma5.bar(dmet.time_normal, av_tot_sample, ls="--",
                                 ec="green", color="lightblue",
                                 alpha=0.8, width=wd/4, align="edge")
                autolabel(S_bar, axis=axma5, fmt='{:.3f}', space=3)
                axma5.set_yticks([])
                # axm5.set_axis_off()
                axma5.set_ylim(bottom=0, top=1.2)
                subplot5_labels += [S_bar[0]]
                labeltext5 += ["Smaple size (g/3h)"]
                # rainfall_amount
            return axma5, subplot5_labels, labeltext5
        def plot_dexcess(subplot5_labels=[], labeltext5=[]):
            axma5_1 = axma5.twinx()

            if dmet.dexcess is not None:
                print("dexcess")
                print(np.shape(dmet.dexcess))
                av_dexcess = np.mean(dmet.dexcess[:, indx[0], indx[1]], axis=1)

                wd = -0.125  # width of 3h x axis found emprically, should heve better automatic

                S_dx = axma5_1.plot(dmet.time_normal, av_dexcess,color="blue")
                # axm5_1.set_ylim(bottom=0, top=
                axma5_1.set_ylabel(' d-excess', color="blue")
                axma5_1.tick_params(axis="y", colors="blue")
                # rainfall_amount
            return axma5_1, subplot5_labels, labeltext5
        def plot_sst(subplot5_labels = [], labeltext5 = []):
            axma5_2 = axma5.twinx()
            if dmet.SST is not None:
                sst_av = np.mean(dmet.SST[:,indx[0], indx[1]],axis=1)
                S_ma= axma5_2.plot(dmet.time_normal, sst_av, color="brown")
                axma5_2.set_ylabel(' SST ', color="brown")
                axma5_2.tick_params(axis="y", colors="brown")
                axma5_2.yaxis.set_label_position("left")
                axma5_2.yaxis.tick_left()

            return axma5_2, subplot5_labels, labeltext5
        axma5, subplot5_labels, labeltext5 = samples(subplot5_labels, labeltext5)
        axma5_1, subplot5_labels, labeltext5 = plot_dexcess(subplot5_labels, labeltext5)
        axma5_2, subplot5_labels, labeltext5 = plot_sst(subplot5_labels, labeltext5)

        axma5_1.legend(subplot5_labels, labeltext5, loc='upper left').set_zorder(99999)

        # FIG2 PLOT6##################################################################################################
        print("#FIG 2 PLOT6##########################################################################################")
        subplot6_labels = []
        labeltext6 = []
        axma6.remove()
        def loc2map(subplot6_labels=[], labeltext6=[]):
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            axma6_0 = figma2.add_subplot(3, 1, 3)
            axma6_0.set_xticks([])
            axma6_0.set_yticks([])
            axma6_0.set_facecolor("lightgray")

            lon0 = dmet.longitude_of_central_meridian_projection_lambert
            lat0 = dmet.latitude_of_projection_origin_projection_lambert
            parallels = dmet.standard_parallel_projection_lambert
            globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
            crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                        globe=globe)
            axma6 = figma2.add_subplot(3, 3, 8, projection=crs)
            axma6.background_patch.set_facecolor('lightskyblue')  # fill_color='lightskyblue'
            axma6.background_patch.set_edgecolor('red')  # fill_color='lightskyblue'
            axma6.background_patch.set_linewidth(5)  # fill_color='lightskyblue'

            axma6.add_feature(cfeature.GSHHSFeature(scale='high'), facecolor='whitesmoke', edgecolor='k', linewidths=2.,
                             zorder=0)
            #CC = axma6.contour(dmet.longitude, dmet.latitude, dmet.land_area_fraction[0, :, :], alpha=0.6, zorder=1,
            #                  levels=[0.9, 1, 1.1],
            #                  colors="b", linewidths=5, transform=ccrs.PlateCarree())
            #CT = axma6.pcolormesh(dmet.longitude, dmet.latitude, dmet.SST[0, :, :],transform=ccrs.PlateCarree(),alpha=0.7)
            mainpoint = axma6.scatter(self.point_lonlat[0], self.point_lonlat[1], c="blue", s=6 ** 2,
                                     transform=ccrs.PlateCarree(), zorder=2)
            allpoint = axma6.scatter(dmet.longitude, dmet.latitude, c="black", s=5 ** 2, transform=ccrs.PlateCarree(),
                                    zorder=1)
            evalpoint = axma6.scatter(dmet.longitude[indx[0], indx[1]], dmet.latitude[indx[0], indx[1]], c="red", s=6 ** 2,
                                     transform=ccrs.PlateCarree(), zorder=4)

            axma6_1 = figma2.add_subplot(3, 3, 9, projection=crs)
            axma6_1.background_patch.set_facecolor('lightskyblue')  # fill_color='lightskyblue'
            axma6_1.add_feature(cfeature.GSHHSFeature(scale='low'), facecolor='whitesmoke', edgecolor='k', linewidths=1.,
                               zorder=0)
            axma6_1.scatter(dmet.x[iindx], dmet.y[jindx], facecolors='none', edgecolors='r', s=6 ** 2, zorder=1,
                           marker='s')

            axma6_1.set_extent([0, 25, 50, 85], crs=ccrs.PlateCarree())
            axma6_1.set_zorder(4)
            from matplotlib.patches import ConnectionPatch
            con = ConnectionPatch(xyA=(dmet.x[iindx], dmet.y[jindx]), xyB=(1, 1), coordsA="data",
                                  coordsB="axes fraction", axesA=axma6_1, axesB=axma6)
            con2 = ConnectionPatch(xyA=(dmet.x[iindx], dmet.y[jindx]), xyB=(1, 0), coordsA="data",
                                   coordsB="axes fraction", axesA=axma6_1, axesB=axma6)
            con.set_color("red")
            axma6_1.add_artist(con)  # axm6_1.transData
            con.set_linewidth(1)
            con2.set_color("red")
            axma6_1.add_artist(con2)  # axm6_1.transData
            con2.set_linewidth(1)
            subplot6_labels += [mainpoint, allpoint, evalpoint]
            labeltext6 += [f"requested pos.\n[{self.point_lonlat[0]},{self.point_lonlat[1]} ]\n {self.point_name}", "model gridpoints", "evaluated gridpoint"]

            axma6_L = figma2.add_subplot(3, 3, 7)
            axma6_L.set_xticks([])
            axma6_L.set_yticks([])
            axma6_L.axis("off")
            # axm6_0.set_zorder(0)
            # axm6.set_zorder(1)
            # axm6_1.set_zorder(2)
            axma6_L.set_zorder(99999)
            # axm6_0.set_facecolor(None)

            return axma6, axma6_1, axma6_0, axma6_L, subplot6_labels, labeltext6
        axma6, axma6_1, axma6_0, axma6_L, subplot6_labels, labeltext6 = loc2map(subplot6_labels, labeltext6)
        axma6_L.legend(subplot6_labels, labeltext6, loc='best').set_zorder(99999)

        #################################
        # SET ADJUSTMENTS ON AXIS
        #################################
        xfmt = mdates.DateFormatter(
            '%d.%m\n%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_maj = mdates.DateFormatter('%d.%m')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute
        xfmt_min = mdates.DateFormatter('%HUTC')  # What format you want on the x-axis. d=day, m=month. H=hour, M=minute

        axma3.xaxis.set_major_locator(mdates.DayLocator())
        axma3.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axma3.xaxis.set_major_formatter(xfmt_maj)
        axma3.xaxis.set_minor_formatter(xfmt_min)
        axma5.xaxis.set_major_locator(mdates.DayLocator())
        axma5.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axma5.xaxis.set_major_formatter(xfmt_maj)
        axma5.xaxis.set_minor_formatter(xfmt_min)
        axma1.xaxis.set_major_locator(mdates.DayLocator())
        axma1.xaxis.set_minor_locator(mdates.HourLocator((0, 6, 12, 18)))
        axma1.xaxis.set_major_formatter(xfmt_maj)
        axma1.xaxis.set_minor_formatter(xfmt_min)
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
        axma5.xaxis.grid(True, which="major", linewidth=2)
        axma5.xaxis.grid(True, which="minor", linestyle="--")

        axma3.tick_params(axis="x", which="major", pad=12,bottom=True,labelbottom=True)
        axma5.tick_params(axis="x", which="major", pad=12,bottom=True,labelbottom=True)
        axma1.tick_params(axis="x", which="major", pad=12,top=True,labeltop=True)
        axma4.tick_params(axis="x", which="major", pad=12,top=True,labeltop=True)

        #figma1.tight_layout()
        #figma2.tight_layout()
        loc_name=self.point_name if self.point_name else self.point_lonlat

        print(dirName_b2 + figname_b2 + "_LOC_loc_name[" + sitename + "]" + ".png")
        figma1.savefig(dirName_b2 + figname_b2 + "_o1_LOC_"+loc_name+"[" + sitename + "].jpg", format="jpg", bbox_inches = "tight", dpi = 200)
        figma2.savefig(dirName_b2 + figname_b2 + "_o2_LOC_"+loc_name+"[" + sitename + "].jpg", format="jpg", bbox_inches = "tight", dpi = 200)

        plt.close()


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

        VM = PMET(date=dt, steps=args.steps, model=args.model, domain_name=args.domain_name,
                      domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, num_point=args.point_num,
                      point_lonlat=args.point_lonlat, point_name=args.point_name)
        VM.retrieve_handler()
        VM.calculations()
        points, indx_sea, indx_land, indx_alldomain, index_neares = VM.points()
        ip = 0
        for po in points:
            jindx, iindx = po
            VM.plot_meteogram(jindx, iindx, dirName_b0, figname_b0, ip, indx_sea, indx_land,indx_alldomain, index_neares)
            plt.close("all")
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
        plt.close("all")

