# %%
# python flexpart_AA.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway
#
import glob
import warnings

# import weathervis.config as wc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import netCDF4 as nc
from add_overlays import *

from weathervis.calculation import *
from weathervis.check_data import *
from weathervis.config import *
from weathervis.domain import *  # require netcdf4
from weathervis.get_data import *
from weathervis.utils import *

print("done")
# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=Downloading)


def flexpart_AA(
    datetime,
    steps=0,
    model="MEPS",
    domain_name=None,
    domain_lonlat=None,
    legend=False,
    info=False,
    save=True,
    grid=True,
):
    print(domain_name)
    for dt in datetime:  # modelrun at time..
        print(dt)
        date = dt[0:-2]
        hour = int(dt[-2:])
        param_sfc = ["air_pressure_at_sea_level", "surface_geopotential"]
        param = param_sfc
        # print(type(steps))
        split = False
        print("\n######## Checking if your request is possible ############")
        try:
            check_all = check_data(
                date=dt, model=model, param=param, p_level=850, step=steps
            )
        except ValueError:
            split = True
            try:
                print("--------> Splitting up your request to find match ############")
                check_sfc = check_data(
                    date=dt, model=model, param=param_sfc, step=steps
                )
                # check_pl = check_data(date=dt, model=model, param=param_pl, p_level=850,step=steps)
            except ValueError:
                print(
                    "!!!!! Sorry this plot is not available for this date. Try with another datetime !!!!!"
                )
                break
        print("--------> Found match for your request ############")

        if not split:
            file_all = check_all.file.loc[0]

            data_domain = domain_input_handler(
                dt,
                model,
                file_all,
                domain_name=domain_name,
                domain_lonlat=domain_lonlat,
            )

            # lonlat = np.array(data_domain.lonlat)
            dmap_meps = get_data(
                model=model,
                data_domain=data_domain,
                param=param,
                file=file_all,
                step=steps,
                date=dt,
                p_level=[850],
            )
            print("\n######## Retrieving data ############")
            print(f"--------> from: {dmap_meps.url} ")
            dmap_meps.retrieve()
            tmap_meps = dmap_meps  # two names for same value, no copying done.
        else:
            # get sfc level data
            file_sfc = check_sfc.file.loc[0]
            data_domain = domain_input_handler(
                dt,
                model,
                file_sfc,
                domain_name=domain_name,
                domain_lonlat=domain_lonlat,
            )
            # lonlat = np.array(data_domain.lonlat)
            dmap_meps = get_data(
                model=model,
                param=param_sfc,
                file=file_sfc,
                step=steps,
                date=dt,
                data_domain=data_domain,
            )
            print("\n######## Retrieving data ############")
            print(f"--------> from: {dmap_meps.url} ")
            dmap_meps.retrieve()

            # get pressure level data
            # file_pl = check_pl.file
            # tmap_meps = get_data(model=model, data_domain=data_domain, param=param_pl, file=file_pl, step=steps, date=dt, p_level = 850)
            # print("\n######## Retrieving data ############")
            # print(f"--------> from: {tmap_meps.url} ")
            # tmap_meps.retrieve()

        # convert fields
        dmap_meps.air_pressure_at_sea_level /= 100

        # read netcdf files with flexpart output
        # all_release_name=['NYAlesund_S1','Tromso_S1', 'cmet1', 'cmet2']
        all_release_name = ["NYAlesund_S1"]
        spec = []
        for release_name in all_release_name:
            path = "/home/centos/flexpart/fp_arome/fp_arome_{0}_forecast_S1.nc".format(
                dt
            )
            findpath = glob.glob(path)
            print(findpath)
            if findpath:
                cdf = nc.Dataset(
                    findpath[0], "r"
                )  # "/home/centos/flexpart/{0}/grid_conc_{1}0000.nc".format(release_name,dt), "r")
                lats = cdf.variables["XLAT"][:]
                lons = cdf.variables["XLONG"][:]
                # lons, lats = np.meshgrid(lons, lats)
                tim = cdf.variables["time"][:]
                levs = cdf.variables["ZTOP"][:]
                print(levs)
                spec1a = cdf.variables["CONC"][:]
                print(np.shape(spec1a))
                print(cdf.NUMRELEASES)
                numr = cdf.NUMRELEASES
                for i in range(0, numr):
                    spec.append(spec1a[:, i, :, :, :].squeeze())
                    print(np.shape(spec))

        print(len(spec))
        print(np.shape(spec))

        # release_name='Tromso_S1'
        # path="/home/centos/flexpart-arome/{0}/{1}*/flexpart_run_d01_combined.nc".format(dt,release_name)
        # findpath= glob.glob(path)
        # print(findpath)
        # cdf = nc.Dataset(findpath[0], "r") #"/home/centos/flexpart/{0}/grid_conc_{1}0000.nc".format(release_name,dt), "r")
        # print("FOOUND")
        # spec1b=cdf.variables["CONC"][:]

        # plot map
        lonlat = [
            dmap_meps.longitude[0, 0],
            dmap_meps.longitude[-1, -1],
            dmap_meps.latitude[0, 0],
            dmap_meps.latitude[-1, -1],
        ]
        print(lonlat)

        lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
        lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
        parallels = dmap_meps.standard_parallel_projection_lambert

        # setting up projection
        globe = ccrs.Globe(
            ellipse="sphere", semimajor_axis=6371000.0, semiminor_axis=6371000.0
        )
        crs = ccrs.LambertConformal(
            central_longitude=lon0,
            central_latitude=lat0,
            standard_parallels=parallels,
            globe=globe,
        )
        # fig1 = plt.figure(figsize=(7, 9))
        make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))

        for tim in np.arange(np.min(steps), np.max(steps) + 1, 1):
            # ax1 = plt.subplot(projection=crs)

            # determine if image should be created for this time step
            stepok = False
            if tim < 25:
                stepok = True
            elif (tim <= 48) and ((tim % 3) == 0):
                stepok = True
            elif (tim <= 66) and ((tim % 6) == 0):
                stepok = True
            if stepok == True:
                l = 0
                last_lvl_idx_for_plotting = np.where(levs > 3000)[0][0]
                print(last_lvl_idx_for_plotting)
                for lev in levs[0 : last_lvl_idx_for_plotting + 1]:
                    print(lev)
                    fig1, ax1 = plt.subplots(
                        1, 1, figsize=(7, 9), subplot_kw={"projection": crs}
                    )
                    ttt = tim
                    tidx = tim - np.min(steps)
                    spec_squeeze = []
                    if lev >= levs[last_lvl_idx_for_plotting]:  # TOC for last levels
                        for i in range(0, len(spec)):
                            ss = spec[i]
                            spec_squeeze.append(np.sum(ss[tim, :, :, :], 0).squeeze())
                            # spec2b=np.sum(spec1b[tim, :, :, :],0).squeeze()
                            lev = 0
                    else:
                        for i in range(0, len(spec)):
                            print(l)
                            ss = spec[i]
                            spec_squeeze.append((ss[tim, l, :, :]).squeeze())
                            # spec2b=(spec1b[tim, l, :, :]).squeeze()
                        l = l + 1

                    for i in range(0, len(spec)):
                        ss = spec_squeeze[i]
                        spec_squeeze[i] = np.where(ss > 1e-8, ss, np.NaN)
                        # spec2b = np.where(spec2b > 1e-8, spec2b, np.NaN)

                    print(
                        "Plotting FLEXPART-AA {0} + {1:02d} UTC, level {2}".format(
                            dt, tim, lev
                        )
                    )
                    # gather, filter and squeeze variables for plotting
                    plev = 0
                    # reduces noise over mountains by removing values over a certain height.
                    Z = dmap_meps.surface_geopotential[tidx, 0, :, :]
                    MSLP = np.where(
                        Z < 30000,
                        dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :],
                        np.NaN,
                    ).squeeze()
                    ax1.contour(
                        dmap_meps.x,
                        dmap_meps.y,
                        MSLP,
                        zorder=0,
                        alpha=0.0,
                        levels=np.arange(960, 1050, 10),
                        colors="white",
                        linewidths=0.5,
                        transform=crs,
                    )
                    my_colors = [
                        plt.cm.Reds,
                        plt.cm.Blues,
                        plt.cm.Greens,
                        plt.cm.Purples,
                        plt.cm.Greys,
                        plt.cm.Oranges,
                    ]
                    print(len(spec))
                    for i in range(0, len(spec)):
                        ss = spec_squeeze[i]
                        if lev >= levs[last_lvl_idx_for_plotting]:
                            ss = np.sum(spec1a[i, tim, :, :, :], 2).squeeze()
                        else:
                            print(np.shape(spec1a))
                            print(i)
                            print(tim)
                            print(l)
                            ss = np.squeeze(spec1a[tim, i, l, :, :])
                        with ax1.hold_limits():
                            F_P = ax1.pcolormesh(
                                lons,
                                lats,
                                ss,
                                norm=colors.LogNorm(vmin=1e-8, vmax=0.1),
                                cmap=my_colors[i],
                                zorder=1,
                                alpha=0.9,
                                transform=ccrs.PlateCarree(),
                            )
                        ##F_P = ax1.pcolormesh(lons, lats, spec2b,  norm=colors.LogNorm(vmin=1e-8, vmax=0.1), cmap=pl, zorder=1, alpha=0.9, transform=ccrs.PlateCarree())
                        # del ss
                    # MSLP with contour labels every 10 hPa
                    C_P = ax1.contour(
                        dmap_meps.x,
                        dmap_meps.y,
                        MSLP,
                        zorder=3,
                        alpha=1.0,
                        levels=np.arange(960, 1050, 1),
                        colors="grey",
                        linewidths=0.5,
                        transform=crs,
                    )
                    C_P = ax1.contour(
                        dmap_meps.x,
                        dmap_meps.y,
                        MSLP,
                        zorder=4,
                        alpha=1.0,
                        levels=np.arange(960, 1050, 10),
                        colors="grey",
                        linewidths=1.0,
                        label="MSLP [hPa]",
                        transform=crs,
                    )
                    ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

                    ax1.add_feature(cfeature.GSHHSFeature(scale="intermediate"))
                    ax1.text(
                        0,
                        1,
                        "{0}_FP_{1}+{2:02d}".format(model, dt, ttt),
                        ha="left",
                        va="bottom",
                        transform=ax1.transAxes,
                        color="black",
                    )

                    legend = False
                    if legend:
                        proxy = [
                            plt.Rectangle(
                                (0, 0),
                                1,
                                1,
                                fc=pc.get_facecolor()[0],
                            )
                            for pc in CF_T.collections
                        ]
                        proxy1 = [
                            plt.axhline(y=0, xmin=1, xmax=1, color="red"),
                            plt.axhline(
                                y=0, xmin=1, xmax=1, color="red", linestyle="dashed"
                            ),
                            plt.axhline(y=0, xmin=1, xmax=1, color="gray"),
                        ]
                        proxy.extend(proxy1)
                        lg = ax1.legend(
                            proxy,
                            [
                                f"RH > 80% [%] at {dmap_meps.pressure[plev]:.0f} hPa",
                                f"T>0 [C] at {dmap_meps.pressure[plev]:.0f} hPa",
                                f"T<0 [C] at {dmap_meps.pressure[plev]:.0f} hPa",
                                "MSLP [hPa]",
                                "",
                            ],
                        )
                        frame = lg.get_frame()
                        frame.set_facecolor("white")
                        frame.set_alpha(1)

                    # if info:
                    #  plt.text(x=0, y=-1, s="INFO: Reduced topographic noise by filtering with surface_geopotential bellow 3000",
                    #           fontsize=7)#, bbox=dict(facecolor='white', alpha=0.5))

                    # lonlat = [dmap_meps.longitude[0, 0], dmap_meps.longitude[-1, -1], dmap_meps.latitude[0, 0],
                    #          dmap_meps.latitude[-1, -1]]
                    # ax.set_extent((lonlat[0]-5, lonlat[1], lonlat[2], lonlat[3]))  # (x0, x1, y0, y1)
                    # ax.set_extent((dmap_meps.x[0], dmap_meps.x[-1], dmap_meps.y[0], dmap_meps.y[-1]))  # (x0, x1, y0, y1)
                    # ax1.set_extent((lonlat[0], lonlat[1], lonlat[2], lonlat[3]))

                    if grid:
                        nicegrid(ax=ax1)

                    add_ISLAS_overlays(ax1)

                    # if domain_name != model and data_domain != None:  # weird bug.. cuts off when sees no data value
                    # ax1.set_extent(lonlat)

                    model = "FLEXPART_AA"
                    print(
                        make_modelrun_folder
                        + "/{0}_{1}_L{2:05.0f}_{3}+{4:02d}.png".format(
                            model, domain_name, lev, dt, tim
                        )
                    )
                    fig1.savefig(
                        make_modelrun_folder
                        + "/{0}_{1}_L{2:05.0f}_{3}+{4:02d}.png".format(
                            model, domain_name, lev, dt, tim
                        ),
                        bbox_inches="tight",
                        dpi=200,
                    )
                    ax1.cla()
                    plt.clf()
                    plt.close(fig1)

    plt.close("all")


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
        default=0,
        nargs="+",
        type=int,
        help="forecast times example --steps 0 3 gives time 0 to 3",
    )
    parser.add_argument("--model", default="MEPS", help="MEPS or AromeArctic")
    parser.add_argument(
        "--domain_name", default=None, help="see domain.py", type=none_or_str
    )
    parser.add_argument(
        "--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]"
    )
    # parser.add_argument("--sim_direction", default="1", help="[ lonmin, lonmax, latmin, latmax]")
    # parser.add_argument("--ZPOIN1_1", default="0", help="[ lonmin, lonmax, latmin, latmax]")
    # parser.add_argument("--ZPOIN1_2", default="0", help="[ lonmin, lonmax, latmin, latmax]")
    parser.add_argument("--legend", default=False, help="Display legend")
    parser.add_argument("--grid", default=True, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    args = parser.parse_args()
    print(args.__dict__)
    flexpart_AA(
        datetime=args.datetime,
        steps=args.steps,
        model=args.model,
        domain_name=args.domain_name,
        domain_lonlat=args.domain_lonlat,
        legend=args.legend,
        info=args.info,
        grid=args.grid,
    )

    # split up in 3 retrievals of up to 24h
    # flexpart_EC(datetime=args.datetime, steps =  [0, np.min([24, np.max(args.steps)])], model = args.model, domain_name = args.domain_name,
    #       domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info, grid=args.grid)
    # if np.max(args.steps)>24:
    #    flexpart_EC(datetime=args.datetime, steps = [27, np.min([36, np.max(args.steps)])], model = args.model, domain_name = args.domain_name,
    #            domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info, grid=args.grid)
    # if np.max(args.steps)>36:
    #    flexpart_EC(datetime=args.datetime, steps = [42, np.max(args.steps)], model = args.model, domain_name = args.domain_name,
    #            domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info, grid=args.grid)
    # datetime, step=4, model= "MEPS", domain = None

# fin
