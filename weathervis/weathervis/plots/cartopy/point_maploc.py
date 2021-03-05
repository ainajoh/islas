from weathervis.config import *
from weathervis.utils import *
from weathervis.domain import *
from weathervis.get_data import *
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
    projectpath = setup_directory(OUTPUTPATH, "{0}".format(modelrun))
    figname = "fc_" + modelrun
    # dirName = projectpath + "result/" + modelrun[0].strftime('%Y/%m/%d/%H/')
    if point_lonlat:
        dirName = projectpath + "meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_lonlat)
    else:
        dirName = projectpath + "meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_name)

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

class MAP():
    def __init__(self, model=None, date=None, steps=[], data = None, domain_name = None, domain_lonlat=None,legend=None,info=None,
                 num_point=None,point_name=None, point_lonlat=None, param_pl=[], param_ml=[],param_sfc=[], param_sfx=[],
                 p_level=None,m_level=None,mbrs=None,url=None):
        self.model =model
        self.date=date
        self.steps=[0] if len(steps) == 0 else steps
        self.data=data
        self.domain_name = domain_name
        self.domain_lonlat = domain_lonlat
        self.num_point = num_point
        self.point_name = point_name
        self.point_lonlat = point_lonlat
        self.param_pl  = param_pl
        self.param_ml  = param_ml
        self.param_sfc = np.unique(["x","y","latitude","land_area_fraction", "surface_geopotential"] + param_sfc).tolist()
        self.param_sfx = param_sfx
        self.param = self.param_ml + self.param_pl + self.param_sfc + self.param_sfx
        self.p_level = p_level
        self.m_level = [0] if m_level is None else m_level
        self.mbrs = mbrs
        self.url = url
        self.point_lonlat = point_lonlat
        self.num_point = num_point
        self.date = date


    def points(self):
        point_lonlat = self.point_lonlat; dmet = self.dmet; num_point = self.num_point
        ind_list = []

        if point_lonlat:
            ind_list = nearest_neighbour(point_lonlat[0], point_lonlat[1], dmet.longitude, dmet.latitude, num_point)
        elif self.point_name:
            sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
            ind_list = nearest_neighbour(sites.loc[self.point_name].lon, sites.loc[self.point_name].lat, dmet.longitude,
                                         dmet.latitude, num_point)
            point_lonlat = [sites.loc[self.point_name].lon, sites.loc[self.point_name].lat]
        poi = ind_list[0:num_point]
        indx_sea = np.where(dmet.land_area_fraction[0][0][:][:] == 0)
        indx_land = np.where(dmet.land_area_fraction[0][0][:][:] == 1)
        indx_alldomain = np.where(dmet.latitude != None)
        ll = np.array([list(item) for item in ind_list[0:num_point]])
        jindx = ll[:, 0]
        iindx = ll[:, 1]
        index_neares = [jindx, iindx]
        return poi, indx_sea, indx_land, indx_alldomain, index_neares

    def retrieve_handler(self):
        dmet_sfx = None;
        dmet_ml = None;
        dmet_pl = None;
        dmet_sfc = None;
        dmet = None
        split = False
        print("\n######## Checking if your request is possible ############")
        try:
            self.param = self.param_pl + self.param_ml + self.param_sfc
            check_all = check_data(date=self.date, model=self.model, param=self.param, step=self.steps,
                                   p_level=self.p_level, m_level=self.m_level,
                                   mbrs=self.mbrs)
        except ValueError:
            split = True
            try:
                print("--------> Splitting up your request to find match ############")
                check_pl = check_data(date=self.date, model=self.model, param=self.param_pl, step=self.steps,
                                      p_level=self.p_level,
                                      mbrs=self.mbrs) if self.param_pl is not None else None
                check_sfc = check_data(date=self.date, model=self.model, param=self.param_sfc, step=self.steps,
                                       mbrs=mbrs) if param_sfc is not None else None
                check_ml = check_data(date=self.date, model=self.model, param=self.param_ml, step=self.steps,
                                      m_level=self.m_level,
                                      mbrs=self.mbrs) if param_ml is not None else None
            except ValueError:
                print("!!!!! Sorry this plot is not availbale for this date. Try with another datetime !!!!!")
                sys.exit(1)
                # break
        print("--------> Found match for your request ############")
        print(check_all.file)
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
            dmet_sfx = get_data(model=self.model, data_domain=data_domain, param=self.param_sfx, file=file_sfx,
                                step=self.steps,
                                date=self.date)
            dmet_sfx.retrieve()

        if not split:
            file_all = check_all.file.loc[0]
            data_domain = domain_input_handler(self.date, self.model, self.domain_name, self.domain_lonlat, file_all)
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
            data_domain = domain_input_handler(dt, model, domain_name, domain_lonlat, file_sfc)
            dmet_sfc = get_data(model=model, param=param_sfc, file=file_sfc, step=steps, date=dt,
                                data_domain=data_domain, mbrs=mbrs)

            file_pl = check_pl.file.loc[0]
            # data_domain = domain_input_handler(dt, model, domain_name, domain_lonlat, file_pl)
            dmet_pl = get_data(model=model, param=param_pl, file=file_pl, step=steps, date=dt,
                               data_domain=data_domain, p_level=p_level, mbrs=mbrs)

            file_ml = check_ml.file.loc[0]
            # data_domain = domain_input_handler(dt, model, domain_name, domain_lonlat, file_ml)
            dmet_ml = get_data(model=model, param=param_ml, file=file_ml, step=steps, date=dt,
                               data_domain=data_domain, m_level=m_level, mbrs=mbrs)

            print("\n######## Retriving data ############")
            print(f"--------> from: {dmet_pl.url} ")
            dmet_pl.retrieve()
            print("\n######## Retriving data ############")
            print(f"--------> from: {dmet_ml.url} ")
            dmet_ml.retrieve()
            print("\n######## Retriving data ############")
            print(f"--------> from: {dmet_sfc.url} ")
            dmet_sfc.retrieve()

        for pm in self.param_sfx:
            setattr(dmet, pm, getattr(dmet_sfx, pm))

        self.dmet = dmet
        self.data_domain = data_domain
        # return dmet, data_domain

    def plot_maplocation_simple(self, dirName_b2, figname_b2, point_lonlat=None):
        dmet = self.dmet
        lon0 = dmet.longitude_of_central_meridian_projection_lambert
        lat0 = dmet.latitude_of_projection_origin_projection_lambert
        parallels = dmet.standard_parallel_projection_lambert
        h_terrain = dmet.surface_geopotential / 9.80665
        globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
        crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                    globe=globe)
        figm2 = plt.figure(figsize=(8, 10), dpi=200)
        ax = figm2.add_subplot(projection=crs)

        ax.background_patch.set_facecolor('lightskyblue')  # fill_color='lightskyblue'

        #lonlat = [dmet.longitude[0, 0], dmet.longitude[-1, -1], dmet.latitude[0, 0],dmet.latitude[-1, -1]]
        ax.add_feature(cfeature.GSHHSFeature(scale='high'), facecolor='whitesmoke', edgecolor='k', linewidths=2.,
                             zorder=2)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
        #ax.set_extent((lonlat[0], lonlat[1], lonlat[2], lonlat[3]))  # (x0, x1, y0, y1)

        plons = [item[0] for item in self.point_lonlat]
        plats = [item[1] for item in self.point_lonlat]
        print(plons)
        print(plats)
        col = ["b","g","r","c","m","y","k","w","lime","orange","darkred","gold","purple"]
        values = np.arange(0,len(col))

        #col = mpl.colors.BASE_COLORS[:len(self.point_name)]
        mainpoint = ax.scatter(plons, plats, c=values[0:len(plons)], s=15**2, transform=ccrs.PlateCarree(),cmap=mpl.colors.ListedColormap(col[0:len(plons)]), zorder=1000, edgecolors="red",linewidths=3)
        plt.legend(handles = mainpoint.legend_elements()[0], labels= self.point_name)

        CC = plt.contour(dmet.longitude, dmet.latitude, dmet.land_area_fraction[0, 0, :, :], alpha=0.0, zorder=2,
                        levels=[0.9, 1, 1.1],
                        colors="b", linewidths=5, transform=ccrs.PlateCarree())
        #h_terrain
        CT = plt.contour(dmet.longitude, dmet.latitude, h_terrain[0,0,:,:] ,levels=np.arange(5,1000,10), alpha=1, zorder=10,
                        colors="b", linewidths=1, transform=ccrs.PlateCarree())
        figm2.tight_layout()
        plt.savefig(dirName_b2 + figname_b2 + ".png", dpi=200)

    def plot_maplocation(self, close2point, dirName_b2, figname_b2, sitename="ALL", point_lonlat=None,
                         all=False):
        dmet = self.dmet
        xy = [dmet.x[0], dmet.x[-1], dmet.y[0], dmet.y[-1]]
        lon0 = dmet.longitude_of_central_meridian_projection_lambert
        lat0 = dmet.latitude_of_projection_origin_projection_lambert
        parallels = dmet.standard_parallel_projection_lambert

        globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
        crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                    globe=globe)
        figm2 = plt.figure(figsize=(7, 7),dpi=200)
        #figm2 = plt.figure(dpi=200)
        ax = figm2.add_subplot(projection=crs)

        ip = 0
        if all == True:
            allpoints = deepcopy(close2point)
            close2point = [0]
        for p in close2point:
            if all == True:
                p = allpoints
            ax.cla()

            ax.background_patch.set_facecolor('lightskyblue')  # fill_color='lightskyblue'

            lonlat = [dmet.longitude[0, 0], dmet.longitude[-1, -1], dmet.latitude[0, 0],
                      dmet.latitude[-1, -1]]
            svalbard_lonlat = [-8, 30, 73, 82]

            if lonlat[0] > svalbard_lonlat[0] and lonlat[1] < svalbard_lonlat[1] and lonlat[2] > svalbard_lonlat[2] and \
                    lonlat[3] < svalbard_lonlat[3]:
                print("svalbard setup")
                file_path = '../../data/shapefiles/svalbard/S100_Land_f_WGS84.shp'  # relative path to the file
                shp = shapereader.Reader(file_path)  # facecolor='whitesmoke', edgecolor='k', linewidths=1., zorder=2
                ax.add_geometries(shp.geometries(), crs=ccrs.PlateCarree(), facecolor='whitesmoke', edgecolor='k',
                                  linewidths=1., zorder=2)  # instead of ax.coastline()
            else:
                ax.add_feature(cfeature.GSHHSFeature(scale='high'), facecolor='whitesmoke', edgecolor='k', linewidths=1.,
                               zorder=2)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
                # ax.coastlines(resolution='10m')
            ax.set_extent((lonlat[0], lonlat[1], lonlat[2], lonlat[3]))  # (x0, x1, y0, y1)

            all_gridpoint = ax.scatter(dmet.longitude, dmet.latitude, s=6.0**2, transform=ccrs.PlateCarree(),
                                    color='k', zorder=4, linestyle='None',edgecolors="k",linewidths=1)

            mainpoint = ax.scatter(point_lonlat[0], point_lonlat[1], s=9.0 ** 2, transform=ccrs.PlateCarree(),
                                       color='lime', zorder=6, linestyle='None', edgecolors="k", linewidths=3)
            closep = ax.scatter(dmet.longitude[p], dmet.latitude[p],s=6.0 ** 2, transform=ccrs.PlateCarree(),
                      color='red', zorder=5, linestyle='None', edgecolors="red", linewidths=1)

            CC = ax.contour(dmet.longitude, dmet.latitude, dmet.land_area_fraction[0, 0, :, :], alpha=0.6, zorder=3,
                             levels=[0.9, 1, 1.1],
                             colors="b", linewidths=5, transform=ccrs.PlateCarree())

            ax.legend((all_gridpoint,mainpoint,closep,CC.collections[0]),
                       ("Gridpoints",f"Point of interest({point_lonlat[1]}N{point_lonlat[0]}E)",
                        "Analysed gridpoints","model coastline"), loc="upper right",ncol=2,mode="expand")
            #gridlines
            #gl = ax.gridlines(crs=crs, linewidth=2, color='gray', alpha=0.5, linestyle='--')

            if all == True:
                figname_b2_2 = figname_b2 + "_[" + sitename + "]"
            else:
                figname_b2_2 = figname_b2 + "_LOC" + str(ip) + \
                               "[" + "{0:.2f}_{1:.2f}]".format(dmet.longitude[p], dmet.latitude[p])

            figm2.tight_layout()
            plt.savefig(dirName_b2 + figname_b2_2 + ".png",dpi=200)
            ip += 1
            plt.close(figm2)

if __name__ == "__main__":
    import argparse
    def none_or_str(value):
        if value == 'None':
            return None
        return value
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
    parser.add_argument("--steps", default=[0], nargs="+", type=int,
                        help="forecast times example --steps 0 3 gives time 0 to 3")
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument("--domain_name", default=None, help="see domain.py", type=none_or_str)
    parser.add_argument("--domain_lonlat", default=None, nargs="+", type=float, help="lonmin lonmax latmin latmax")
    parser.add_argument("--point_name", default=None, nargs="+", help="see sites.csv")
    parser.add_argument("--point_lonlat", default=None, nargs="+", type=float, help="lon lat")
    parser.add_argument("--point_num", default=1, type=int)
    parser.add_argument("--plot", default="all", help="Display legend")
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()

    for dt in args.datetime:
        dirName_b0, dirName_b1, dirName_b2, dirName_b3, figname_b0, figname_b1, figname_b2, figname_b3 = setup_met_directory(
            dt, args.point_name, args.point_lonlat)

        VM = MAP(date=dt, steps=args.steps, model=args.model, domain_name=args.domain_name,
                      domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, num_point=args.point_num,
                      point_lonlat=args.point_lonlat, point_name=args.point_name)

        VM.retrieve_handler()
        if args.point_lonlat is None and args.point_name is not None:
            point_lonlat = []
            for pname in args.point_name:
                sites = pd.read_csv("../../data/sites.csv", sep=";", header=0, index_col=0)
                lonlat = [sites.loc[pname].lon, sites.loc[pname].lat]
                point_lonlat += [lonlat]
        VM.point_lonlat = point_lonlat
        #points, indx_sea, indx_land, indx_alldomain, index_neares = VM.points()
        VM.plot_maplocation_simple( dirName_b2, figname_b2)
        #for po in points:
        #    jindx, iindx = po
        #    VM.plot_meteogram(jindx, iindx, dirName_b0, figname_b0, ip)
        #    ip += 1

        #averagesite = ["ALL_DOMAIN", "ALL_NEAREST", "LAND", "SEA"]  # "ALL_NEAREST", "LAND", "SEA",

        #plot_maplocation(dmet, data_domain, indx_sea, dirName_b2, figname_b2, sitename, point_lonlat, all=True)

