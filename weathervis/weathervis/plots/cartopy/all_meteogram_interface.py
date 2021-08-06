from weathervis.config import *
from weathervis.utils import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.calculation import *
import os
from point_meteogram import *
from point_vertical_meteogram import *
from point_maploc import *
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
from cartopy.io import (
    shapereader,
)  # For reading shapefiles containg high-resolution coastline.
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
            print(
                f"\n####### Setting up domain for coordinates: {domain_lonlat} ##########"
            )
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
    # projectpath = setup_directory(OUTPUTPATH, "{0}".format(dt))
    projectpath = OUTPUTPATH + "{0}".format(dt)
    figname = "fc_" + modelrun
    # dirName = projectpath + "result/" + modelrun[0].strftime('%Y/%m/%d/%H/')
    if point_lonlat:
        dirName = (
            projectpath + "meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_lonlat)
        )
    else:
        dirName = (
            projectpath + "meteogram/" + "fc_" + modelrun[:-2] + "/" + str(point_name)
        )

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
    return (
        dirName_b0,
        dirName_b1,
        dirName_b2,
        dirName_b3,
        figname_b0,
        figname_b1,
        figname_b2,
        figname_b3,
    )


class LOC_PLOTS:
    def __init__(
        self,
        model,
        date,
        steps,
        data=None,
        domain_name=None,
        domain_lonlat=None,
        legend=None,
        info=None,
        num_point=None,
        point_name=None,
        point_lonlat=None,
        param_pl=[],
        param_ml=[],
        param_sfc=[],
        param_sfx=[],
    ):
        self.model = model
        self.date = date
        self.steps = steps
        self.data = data
        self.domain_name = domain_name
        self.domain_lonlat = domain_lonlat
        self.num_point = num_point
        self.point_name = point_name
        self.point_lonlat = point_lonlat
        self.param_pl = param_pl
        self.param_ml = param_ml
        self.param_sfc = param_sfc
        self.param_sfx = param_sfx
        self.param = self.param_ml + self.param_pl + self.param_sfc + self.param_sfx
        self.p_level = None
        self.m_level = None
        self.mbrs = None
        self.url = None
        self.point_lonlat = point_lonlat
        self.num_point = num_point
        self.date = date

    def points(self, dmet):
        point_lonlat = self.point_lonlat
        dmet = self.dmet
        num_point = self.num_point
        ind_list = []

        if point_lonlat:
            ind_list = nearest_neighbour(
                point_lonlat[0],
                point_lonlat[1],
                dmet.longitude,
                dmet.latitude,
                num_point,
            )
        elif self.point_name:

            point = setup_site(self.point_name)
            ind_list = nearest_neighbour(
                point["lon"], point["lat"], dmet.longitude, dmet.latitude, num_point
            )
            point_lonlat = [point["lon"], point["lat"]]
            self.point_lonlat = point_lonlat

        poi = ind_list[0:num_point]
        indx_sea = np.where(dmet.land_area_fraction[0][0][:][:] == 0)
        indx_land = np.where(dmet.land_area_fraction[0][0][:][:] == 1)
        indx_alldomain = np.where(dmet.latitude != None)
        ll = np.array([list(item) for item in ind_list[0:num_point]])
        jindx = ll[:, 0]
        iindx = ll[:, 1]
        index_neares = [jindx, iindx]
        return poi, indx_sea, indx_land, indx_alldomain, index_neares, self.point_lonlat

    def retrieve_handler(self):

        print("\n######## Checking if your request is possible ############")
        self.param = self.param_pl + self.param_ml + self.param_sfc + self.param_sfx
        dmet, data_domain, bad_param = checkget_data_handler(
            all_param=self.param,
            date=self.date,
            model=self.model,
            step=self.steps,
            p_level=self.p_level,
            m_level=self.m_level,
            mbrs=self.mbrs,
            domain_name=self.domain_name,
            domain_lonlat=self.domain_lonlat,
        )

        self.dmet = dmet
        self.data_domain = data_domain
        print("DATA RETRIEVED")

        return dmet, data_domain, bad_param


def input_handler(
    date,
    steps,
    model,
    domain_name,
    domain_lonlat,
    legend,
    info,
    num_point,
    point_lonlat,
    point_name,
):
    (
        dirName_b0,
        dirName_b1,
        dirName_b2,
        dirName_b3,
        figname_b0,
        figname_b1,
        figname_b2,
        figname_b3,
    ) = setup_directory(dt, point_name, point_lonlat)

    PM = PMET(
        date=date,
        steps=steps,
        model=model,
        domain_name=domain_name,
        domain_lonlat=domain_lonlat,
        legend=legend,
        info=info,
        num_point=num_point,
        point_lonlat=point_lonlat,
        point_name=point_name,
    )

    param_pl = PM.param_pl
    param_ml = PM.param_ml
    param_sfc = PM.param_sfc
    param_sfx = PM.param_sfx

    VM = VERT_MET(
        date=date,
        steps=steps,
        model=model,
        domain_name=domain_name,
        domain_lonlat=domain_lonlat,
        legend=legend,
        info=info,
        num_point=num_point,
        point_lonlat=point_lonlat,
        point_name=point_name,
    )

    param_pl = VM.param_pl + param_pl
    param_ml = VM.param_ml + param_ml
    param_sfc = VM.param_sfc + param_sfc
    param_sfx = VM.param_sfx + param_sfx

    param = param_pl + param_ml + param_sfc + param_sfx

    LP = LOC_PLOTS(
        date=date,
        steps=steps,
        model=model,
        domain_name=domain_name,
        domain_lonlat=domain_lonlat,
        legend=legend,
        info=info,
        num_point=num_point,
        point_lonlat=point_lonlat,
        point_name=point_name,
        param_pl=param_pl,
        param_ml=param_ml,
        param_sfc=param_sfc,
        param_sfx=param_sfx,
    )

    dmet, data_domain, bad_param = LP.retrieve_handler()
    print("bad_param")
    print(bad_param)
    # for bpam in bad_param:
    #    #a = np.full([height, width, 9], np.nan)
    #    setattr(dmet,bpam, np.nan)

    LP.dmet = dmet
    points, indx_sea, indx_land, indx_alldomain, index_neares, point_lonlat = LP.points(
        dmet
    )

    PM.dmet = dmet
    PM.calculations()

    VM.dmet = dmet
    print("start cald")
    VM.calculations()
    print("end cald")
    M = MAP(point_lonlat=point_lonlat, point_name=point_name)
    M.dmet = dmet

    # POINT PLOTS
    M.plot_maplocation(points, dirName_b2, figname_b2, point_lonlat=point_lonlat)

    ip = 0
    for po in points:
        jindx, iindx = po
        PM.plot_meteogram(jindx, iindx, dirName_b0, figname_b0, ip)
        VM.vertical_met(jindx, iindx, dirName_b1, figname_b1, ip, p_top=500)
        ip += 1

    # AVERAGE PLOTS
    averagesite = ["ALL_DOMAIN", "ALL_NEAREST", "LAND", "SEA"]
    for av in averagesite:
        indx = indx_alldomain if av == "ALL_DOMAIN" else None
        indx = indx_sea if av == "SEA" else indx
        indx = indx_land if av == "LAND" else indx
        indx = index_neares if av == "ALL_NEAREST" else indx
        PM.meteogram_average(indx, dirName_b3, figname_b3, av)
        M.plot_maplocation(
            indx,
            dirName_b2,
            figname_b2,
            sitename=av,
            point_lonlat=point_lonlat,
            all=True,
        )


if __name__ == "__main__":
    import argparse

    def none_or_str(value):
        if value == "None":
            return None
        return value

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+"
    )
    parser.add_argument(
        "--steps",
        default=[0, 10],
        nargs="+",
        type=int,
        help="forecast times example --steps 0 3 gives time 0 to 3",
    )
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument(
        "--domain_name", default=None, help="see domain.py", type=none_or_str
    )
    parser.add_argument(
        "--domain_lonlat",
        default=None,
        nargs="+",
        type=float,
        help="lonmin lonmax latmin latmax",
    )
    parser.add_argument("--point_name", default=None, help="see sites.yaml")
    parser.add_argument(
        "--point_lonlat", default=None, nargs="+", type=float, help="lon lat"
    )
    parser.add_argument("--point_num", default=1, type=int)
    parser.add_argument("--plot", default="all", help="Display legend")
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()

    for dt in args.datetime:
        # print(args.datetime)
        input_handler(
            date=dt,
            steps=args.steps,
            model=args.model,
            domain_name=args.domain_name,
            domain_lonlat=args.domain_lonlat,
            legend=args.legend,
            info=args.info,
            num_point=args.point_num,
            point_lonlat=args.point_lonlat,
            point_name=args.point_name,
        )
